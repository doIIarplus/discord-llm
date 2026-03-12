"""!time command - converts human-readable time to Discord timestamps."""

import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import discord

# Map role names to IANA timezone identifiers
TIMEZONE_ROLES = {
    'PST': 'America/Los_Angeles',
    'PDT': 'America/Los_Angeles',
    'PT': 'America/Los_Angeles',
    'MST': 'America/Denver',
    'MDT': 'America/Denver',
    'MT': 'America/Denver',
    'CST': 'America/Chicago',
    'CDT': 'America/Chicago',
    'CT': 'America/Chicago',
    'EST': 'America/New_York',
    'EDT': 'America/New_York',
    'ET': 'America/New_York',
    'AKST': 'America/Anchorage',
    'AKDT': 'America/Anchorage',
    'HST': 'Pacific/Honolulu',
    'GMT': 'Europe/London',
    'BST': 'Europe/London',
    'CET': 'Europe/Berlin',
    'CEST': 'Europe/Berlin',
    'EET': 'Europe/Helsinki',
    'EEST': 'Europe/Helsinki',
    'JST': 'Asia/Tokyo',
    'KST': 'Asia/Seoul',
    'AEST': 'Australia/Sydney',
    'AEDT': 'Australia/Sydney',
    'ACST': 'Australia/Adelaide',
    'ACDT': 'Australia/Adelaide',
    'AWST': 'Australia/Perth',
    'NZST': 'Pacific/Auckland',
    'NZDT': 'Pacific/Auckland',
    'IST': 'Asia/Kolkata',
    'SGT': 'Asia/Singapore',
    'HKT': 'Asia/Hong_Kong',
    'PHT': 'Asia/Manila',
    'ICT': 'Asia/Bangkok',
    'WIB': 'Asia/Jakarta',
}

DAY_NAMES = {
    'monday': 0, 'mon': 0,
    'tuesday': 1, 'tue': 1, 'tues': 1,
    'wednesday': 2, 'wed': 2,
    'thursday': 3, 'thu': 3, 'thur': 3, 'thurs': 3,
    'friday': 4, 'fri': 4,
    'saturday': 5, 'sat': 5,
    'sunday': 6, 'sun': 6,
}

RELATIVE_DAYS = {
    'today': 0,
    'tomorrow': 1,
    'tmrw': 1,
    'tmr': 1,
}


def get_user_timezone(member: discord.Member) -> ZoneInfo:
    """Check member roles for a timezone role, default to UTC."""
    for role in member.roles:
        role_name = role.name.upper().strip()
        if role_name in TIMEZONE_ROLES:
            return ZoneInfo(TIMEZONE_ROLES[role_name])
    return ZoneInfo('UTC')


def parse_time_input(text: str, tz: ZoneInfo) -> datetime | None:
    """Parse human-readable time string into a datetime in the given timezone.

    Supports formats like:
      - "sunday 9am", "fri 5:30pm", "tomorrow 3pm"
      - "march 15 2pm", "3/15 2pm", "2026-03-15 14:00"
      - "in 2 hours", "in 30 minutes", "in 3 days"
      - "9pm" (assumes today, or tomorrow if already passed)
      - "noon", "midnight"
    """
    text = text.strip().lower()
    now = datetime.now(tz)

    # Handle "in X hours/minutes/days"
    relative_match = re.match(
        r'in\s+(\d+)\s*(h(?:ou)?rs?|m(?:in(?:ute)?s?)?|d(?:ays?)?|w(?:eeks?)?)',
        text
    )
    if relative_match:
        amount = int(relative_match.group(1))
        unit = relative_match.group(2)[0]
        if unit == 'h':
            return now + timedelta(hours=amount)
        elif unit == 'm':
            return now + timedelta(minutes=amount)
        elif unit == 'd':
            return now + timedelta(days=amount)
        elif unit == 'w':
            return now + timedelta(weeks=amount)

    # Extract time component
    time_hour, time_minute = None, 0

    # Check for "noon" / "midnight"
    if 'noon' in text:
        time_hour = 12
        text = text.replace('noon', '').strip()
    elif 'midnight' in text:
        time_hour = 0
        text = text.replace('midnight', '').strip()

    if time_hour is None:
        # Match "9am", "9:30pm", "14:00", "9 am", "9:30 pm"
        time_match = re.search(r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)?', text)
        if time_match:
            time_hour = int(time_match.group(1))
            time_minute = int(time_match.group(2)) if time_match.group(2) else 0
            ampm = time_match.group(3)
            if ampm == 'pm' and time_hour != 12:
                time_hour += 12
            elif ampm == 'am' and time_hour == 12:
                time_hour = 0
            # Remove the time part for date parsing
            text = text[:time_match.start()].strip() + ' ' + text[time_match.end():].strip()
            text = text.strip()

    if time_hour is None:
        # No time found at all
        return None

    # Now parse the date component from remaining text
    target_date = None

    # Check for day names (sunday, mon, etc.)
    for name, weekday in DAY_NAMES.items():
        if name in text.split() or text.startswith(name):
            days_ahead = weekday - now.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            target_date = now.date() + timedelta(days=days_ahead)
            break

    # Check for relative days (today, tomorrow)
    if target_date is None:
        for name, offset in RELATIVE_DAYS.items():
            if name in text.split() or text == name:
                target_date = now.date() + timedelta(days=offset)
                break

    # Check for month day format: "march 15", "mar 15"
    if target_date is None:
        month_names = {
            'jan': 1, 'january': 1, 'feb': 2, 'february': 2,
            'mar': 3, 'march': 3, 'apr': 4, 'april': 4,
            'may': 5, 'jun': 6, 'june': 6,
            'jul': 7, 'july': 7, 'aug': 8, 'august': 8,
            'sep': 9, 'september': 9, 'oct': 10, 'october': 10,
            'nov': 11, 'november': 11, 'dec': 12, 'december': 12,
        }
        for name, month in month_names.items():
            pattern = rf'\b{name}\s+(\d{{1,2}})\b'
            m = re.search(pattern, text)
            if m:
                day = int(m.group(1))
                year = now.year
                try:
                    target_date = datetime(year, month, day).date()
                    # If the date is in the past, use next year
                    if target_date < now.date():
                        target_date = datetime(year + 1, month, day).date()
                except ValueError:
                    return None
                break

    # Check for numeric date: "3/15", "03/15"
    if target_date is None:
        date_match = re.search(r'(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?', text)
        if date_match:
            month = int(date_match.group(1))
            day = int(date_match.group(2))
            year = int(date_match.group(3)) if date_match.group(3) else now.year
            if year < 100:
                year += 2000
            try:
                target_date = datetime(year, month, day).date()
                if target_date < now.date() and not date_match.group(3):
                    target_date = datetime(year + 1, month, day).date()
            except ValueError:
                return None

    # No date specified - assume today, or tomorrow if time already passed
    if target_date is None:
        target_date = now.date()
        candidate = datetime(
            target_date.year, target_date.month, target_date.day,
            time_hour, time_minute, tzinfo=tz
        )
        if candidate <= now:
            target_date += timedelta(days=1)

    return datetime(
        target_date.year, target_date.month, target_date.day,
        time_hour, time_minute, tzinfo=tz
    )


def format_discord_timestamp(dt: datetime) -> str:
    """Format a datetime as Discord timestamp strings."""
    unix = int(dt.timestamp())
    return f"<t:{unix}:f> (<t:{unix}:R>)"


async def handle_time_command(message: discord.Message, time_text: str) -> None:
    """Handle the !time command."""
    if not time_text:
        await message.channel.send(
            "usage: `!time <human time>`\n"
            "examples: `!time sunday 9am`, `!time tomorrow 3pm`, `!time in 2 hours`, `!time march 15 noon`\n"
            "i'll check your roles for a timezone (PST, EST, etc). if none found, i assume UTC."
        )
        return

    tz = get_user_timezone(message.author)
    tz_label = str(tz)
    # Show a friendlier label
    for role_name, iana in TIMEZONE_ROLES.items():
        if iana == str(tz):
            tz_label = role_name
            break

    dt = parse_time_input(time_text, tz)
    if dt is None:
        await message.channel.send(
            f"couldn't parse \"{time_text}\" — try something like `sunday 9am`, `tomorrow 3pm`, or `in 2 hours`"
        )
        return

    timestamp_str = format_discord_timestamp(dt)
    if tz == ZoneInfo('UTC'):
        tz_note = "(no timezone role found, using UTC)"
    else:
        tz_note = f"(using {tz_label} from your roles)"

    await message.channel.send(f"{timestamp_str}\n{tz_note}")
