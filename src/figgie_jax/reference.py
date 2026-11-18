"""Independent, uncompiled Python oracle for economic transition conformance.

This module uses ordinary lists and scalar branches. It does not call JAX game
logic. The scheduler is tested separately; oracle inputs are timestamped events.
"""

from copy import deepcopy

from figgie_jax.types import Config, Kind, Reason


def transition(
    state: dict, player: int, action: tuple[int, int, int, int], at_ms: int, config: Config
) -> tuple[dict, int, bool, bool]:
    """Apply an event to a plain snapshot and return state, reason, accepted, traded."""
    s = deepcopy(state)
    if s["done"]:
        return s, Reason.ROUND_OVER, False, False
    if at_ms < s["time_ms"]:
        return s, Reason.TIME_REVERSED, False, False
    if at_ms >= config.duration_ms:
        counts = [hand[s["goal"]] for hand in s["hands"]]
        largest = max(counts)
        winners = sum(x == largest for x in counts)
        bonus = (200 - 10 * sum(counts)) / winners
        for i, count in enumerate(counts):
            s["cash"][i] += 10 * count + (bonus if count == largest else 0)
        s.update(
            time_ms=config.duration_ms,
            done=True,
            prices=[[0] * 4 for _ in range(2)],
            owners=[[-1] * 4 for _ in range(2)],
            quote_ids=[[0] * 4 for _ in range(2)],
        )
        return s, Reason.ROUND_OVER, False, False
    s["time_ms"] = at_ms
    kind, suit, price, quote_id = action
    accepted = traded = False
    reason = Reason.INVALID
    if 0 <= player < config.num_players:
        if kind == Kind.WAIT:
            accepted = True
        elif 0 <= suit < 4:
            side = int(kind in (Kind.ASK, Kind.BUY, Kind.CANCEL_ASK))
            owner = s["owners"][side][suit]
            old_price = s["prices"][side][suit]
            if kind in (Kind.BID, Kind.ASK):
                improves = price > old_price if kind == Kind.BID else price < old_price
                accepted = 1 <= price <= config.max_price and (owner in (-1, player) or improves)
                if kind == Kind.ASK:
                    accepted &= s["hands"][player][suit] > 0
                elif not config.allow_negative_cash:
                    accepted &= s["cash"][player] >= price
                if accepted:
                    s["prices"][side][suit] = price
                    s["owners"][side][suit] = player
                    s["quote_ids"][side][suit] = s["events"] + 1
            elif kind in (Kind.BUY, Kind.SELL):
                fresh = quote_id > 0 and quote_id == s["quote_ids"][side][suit]
                buyer, seller = (player, owner) if kind == Kind.BUY else (owner, player)
                accepted = owner >= 0 and owner != player and fresh
                if accepted:
                    accepted &= s["hands"][seller][suit] > 0
                    if not config.allow_negative_cash:
                        accepted &= s["cash"][buyer] >= old_price
                if not fresh:
                    reason = Reason.STALE_QUOTE
                if accepted:
                    s["hands"][buyer][suit] += 1
                    s["hands"][seller][suit] -= 1
                    s["cash"][buyer] -= old_price
                    s["cash"][seller] += old_price
                    row = [at_ms, buyer, seller, suit, old_price]
                    s["history"][s["trade_count"] % config.history_size] = row
                    s["trade_count"] += 1
                    s["prices"] = [[0] * 4 for _ in range(2)]
                    s["owners"] = [[-1] * 4 for _ in range(2)]
                    s["quote_ids"] = [[0] * 4 for _ in range(2)]
                    traded = True
            elif kind in (Kind.CANCEL_BID, Kind.CANCEL_ASK) and owner == player:
                accepted = True
                s["prices"][side][suit] = 0
                s["owners"][side][suit] = -1
                s["quote_ids"][side][suit] = 0
    if accepted:
        reason = Reason.ACCEPTED
    s["events"] += 1
    if s["events"] >= config.max_events:
        s["done"] = s["truncated"] = True
        reason = Reason.EVENT_LIMIT
    return s, reason, bool(accepted), traded
