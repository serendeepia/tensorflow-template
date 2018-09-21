import re
from datetime import timedelta

MINUTE_SECONDS = 60
HOUR_SECONDS = 60 * MINUTE_SECONDS
DAY_SECONDS = 24 * HOUR_SECONDS
WEEK_SECONDS = 7 * DAY_SECONDS


def tdelta(input):
    keys = ["weeks", "days", "hours", "minutes", "seconds"]
    regex = "".join(
            ["\s*((?P<%s>\d+)\s*(%s(%s(s)?)?)(\s*(and|,)?\s*)?)?" % (k, k[0], k[1:-1]) for k in
             keys])
    kwargs = {}
    for k, v in re.match(regex, input).groupdict(default="0").items():
        kwargs[k] = int(v)
    return timedelta(**kwargs)


def format_elapsed(seconds):
    weeks = int(seconds / WEEK_SECONDS)
    seconds_remaining = seconds - weeks * WEEK_SECONDS
    days = int(seconds_remaining / DAY_SECONDS)
    seconds_remaining = seconds_remaining - days * DAY_SECONDS
    hours = int(seconds_remaining / HOUR_SECONDS)
    seconds_remaining = seconds_remaining - hours * HOUR_SECONDS
    minutes = int(seconds_remaining / MINUTE_SECONDS)
    seconds_remaining = seconds_remaining - minutes * MINUTE_SECONDS
    seconds_remaining = int(seconds_remaining)
    if weeks > 0:
        return '{}w {}d'.format(weeks, days)
    elif days > 0:
        return '{}d {}h'.format(days, hours)
    elif hours > 0:
        return '{}h {}m'.format(hours, minutes)
    elif minutes > 0:
        return '{}m'.format(minutes)
    else:
        return '{}s'.format(seconds_remaining)
