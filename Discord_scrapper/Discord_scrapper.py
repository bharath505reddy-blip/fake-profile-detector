import os
import csv
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

import discord
from discord.ext import commands

# Load .env file from the same directory as this script (if it exists)
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _key, _, _val = _line.partition("=")
            os.environ.setdefault(_key.strip(), _val.strip())

TOKEN = os.getenv("DISCORD_BOT_TOKEN")

# ========= CONFIG (tweak these) =========
MIN_ACCOUNT_AGE_DAYS = 7          # new accounts are riskier
JOIN_BURST_WINDOW_SEC = 120       # watch joins in last 2 minutes
JOIN_BURST_THRESHOLD = 5          # 5+ joins in 2 min => suspicious server activity
MESSAGE_BURST_WINDOW_SEC = 60     # messages in last 60s
MESSAGE_BURST_THRESHOLD = 8       # 8+ msgs/min soon after join

LOG_CSV = Path(__file__).parent / "discord_profile_signals.csv"
# =======================================

@dataclass
class UserSignals:
    timestamp_utc: str
    guild_id: str
    guild_name: str
    user_id: str
    username: str
    created_at_utc: str
    joined_at_utc: str
    account_age_days: float
    has_avatar: int
    suspicion_score: int
    reasons: str
    action_taken: str


def utc_now_str() -> str:
    return datetime.now(timezone.utc).isoformat()

def dt_to_str(dt: datetime | None) -> str:
    if not dt:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()

def ensure_csv_header():
    if not LOG_CSV.exists():
        with open(LOG_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=UserSignals.__annotations__.keys())
            writer.writeheader()

def append_csv(row: UserSignals):
    ensure_csv_header()
    with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=UserSignals.__annotations__.keys())
        writer.writerow(asdict(row))


# Intents:
# - members is needed to get join events + member data
# - message_content is OPTIONAL (only if you want message burst tracking)
intents = discord.Intents.default()
intents.members = True
intents.message_content = False  # set True ONLY if you enabled Message Content Intent in Dev Portal

bot = commands.Bot(command_prefix="!", intents=intents)

# In-memory trackers
recent_joins: dict[int, list[float]] = {}         # guild_id -> list of join timestamps (epoch seconds)
recent_messages: dict[tuple[int, int], list[float]] = {}  # (guild_id, user_id) -> message timestamps
# Tracks (guild_id, user_id) pairs whose message burst has already been logged,
# so we only write one CSV row per burst session instead of one per message.
_burst_logged: set[tuple[int, int]] = set()


def compute_suspicion(member: discord.Member, join_burst: bool, msg_burst: bool) -> tuple[int, list[str]]:
    reasons = []
    score = 0

    created = member.created_at
    joined = member.joined_at or datetime.now(timezone.utc)

    account_age_days = (joined - created).total_seconds() / 86400.0

    # 1) New account
    if account_age_days < MIN_ACCOUNT_AGE_DAYS:
        score += 40
        reasons.append(f"new_account<{MIN_ACCOUNT_AGE_DAYS}d")

    # 2) Default avatar / no custom avatar
    if member.avatar is None:
        score += 15
        reasons.append("no_custom_avatar")

    # 3) Server join burst happening now (raid-like)
    if join_burst:
        score += 25
        reasons.append(f"join_burst>{JOIN_BURST_THRESHOLD}/{JOIN_BURST_WINDOW_SEC}s")

    # 4) Optional: message burst soon after join
    if msg_burst:
        score += 30
        reasons.append(f"message_burst>{MESSAGE_BURST_THRESHOLD}/{MESSAGE_BURST_WINDOW_SEC}s")

    return score, reasons


def is_join_burst(guild_id: int) -> bool:
    now = time.time()
    lst = recent_joins.get(guild_id, [])
    lst = [t for t in lst if now - t <= JOIN_BURST_WINDOW_SEC]
    recent_joins[guild_id] = lst
    return len(lst) >= JOIN_BURST_THRESHOLD


def is_message_burst(guild_id: int, user_id: int) -> bool:
    now = time.time()
    key = (guild_id, user_id)
    lst = recent_messages.get(key, [])
    lst = [t for t in lst if now - t <= MESSAGE_BURST_WINDOW_SEC]
    recent_messages[key] = lst
    return len(lst) >= MESSAGE_BURST_THRESHOLD


def _make_signals_row(member: discord.Member, score: int, reasons: list[str], action_taken: str) -> UserSignals:
    """Build a UserSignals dataclass from a guild member."""
    guild = member.guild
    age_days = (
        round((member.joined_at - member.created_at).total_seconds() / 86400.0, 4)
        if member.joined_at else -1
    )
    return UserSignals(
        timestamp_utc=utc_now_str(),
        guild_id=str(guild.id),
        guild_name=guild.name,
        user_id=str(member.id),
        username=(
            f"{member.name}#{member.discriminator}"
            if member.discriminator != "0" else member.name
        ),
        created_at_utc=dt_to_str(member.created_at),
        joined_at_utc=dt_to_str(member.joined_at),
        account_age_days=age_days,
        has_avatar=0 if member.avatar is None else 1,
        suspicion_score=score,
        reasons="|".join(reasons),
        action_taken=action_taken,
    )


@bot.event
async def on_ready():
    print(f"Logged in as {bot.user} | message_content_intent={bot.intents.message_content}")


@bot.event
async def on_member_join(member: discord.Member):
    guild = member.guild
    now = time.time()

    # track joins
    recent_joins.setdefault(guild.id, []).append(now)
    join_burst = is_join_burst(guild.id)

    # message burst unknown at join time
    suspicion_score, reasons = compute_suspicion(member, join_burst=join_burst, msg_burst=False)

    action_taken = "none"

    if suspicion_score >= 70:
        try:
            until = datetime.now(timezone.utc) + timedelta(minutes=10)
            await member.timeout(until, reason="Auto-flag: suspected fake/spam account")
            action_taken = "timeout_10m"
        except discord.Forbidden:
            action_taken = "timeout_failed_no_perms"
        except discord.HTTPException as exc:
            action_taken = f"timeout_failed_http_{exc.status}"

    append_csv(_make_signals_row(member, suspicion_score, reasons, action_taken))

    log_channel = discord.utils.get(guild.text_channels, name="mod-log")
    if log_channel:
        await log_channel.send(
            f"🚩 **Join check:** {member.mention} | score={suspicion_score} | "
            f"reasons: `{', '.join(reasons) or 'none'}` | action={action_taken}"
        )


@bot.event
async def on_message(message: discord.Message):
    # Only works for others' messages if you enabled Message Content Intent.
    if message.author.bot or not message.guild:
        return

    # track message timestamps (no content stored)
    key = (message.guild.id, message.author.id)
    recent_messages.setdefault(key, []).append(time.time())

    if bot.intents.message_content:
        member = message.guild.get_member(message.author.id)
        if member and member.joined_at:
            time_since_join = (datetime.now(timezone.utc) - member.joined_at).total_seconds()

            if time_since_join <= 900:
                # Only log burst once per join session — not on every message after threshold
                if key not in _burst_logged and is_message_burst(message.guild.id, message.author.id):
                    _burst_logged.add(key)
                    join_burst = is_join_burst(message.guild.id)
                    suspicion_score, reasons = compute_suspicion(member, join_burst=join_burst, msg_burst=True)
                    append_csv(_make_signals_row(member, suspicion_score, reasons, "none"))
            else:
                # Past the 15-minute window — clean up this user's tracking data
                _burst_logged.discard(key)
                recent_messages.pop(key, None)

    await bot.process_commands(message)


@bot.command()
@commands.has_permissions(moderate_members=True)
async def score(ctx: commands.Context, member: discord.Member):
    """Manual check: !score @user"""
    join_burst = is_join_burst(ctx.guild.id)
    msg_burst = is_message_burst(ctx.guild.id, member.id) if bot.intents.message_content else False
    suspicion_score, reasons = compute_suspicion(member, join_burst=join_burst, msg_burst=msg_burst)

    await ctx.send(
        f"🔎 {member.mention} score={suspicion_score} | reasons: `{', '.join(reasons) or 'none'}`"
    )


if not TOKEN:
    raise SystemExit("Set DISCORD_BOT_TOKEN environment variable first.")

bot.run(TOKEN)
