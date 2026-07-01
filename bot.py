import asyncio
import io
import logging
import os
import re
import sys
import tempfile

import aiohttp
import discord
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from memory import add_memory, get_relevant_memories, delete_user_memories, get_memory_count
from video import create_pexels_video

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("echo")

# Ensure UTF-8 output on Windows
sys.stdout.reconfigure(encoding="utf-8")

# ── Persona ───────────────────────────────────────────────────────────────────

with open("chat.txt", "r", encoding="utf-8") as f:
    raw = f.read().strip()
    # Strip the wrapping list brackets if present in chat.txt
    PERSONA = raw.strip("[]")

# ── Environment ───────────────────────────────────────────────────────────────

load_dotenv()
DISCORD_TOKEN = os.getenv("Discord_Token")
HF_TOKEN      = os.getenv("HF_TOKEN")
PEXELS_API    = os.getenv("PEXELS_API")

if not DISCORD_TOKEN or not HF_TOKEN:
    raise ValueError("Discord_Token and HF_TOKEN must be set in .env")

# ── AI client ─────────────────────────────────────────────────────────────────

client_hf = InferenceClient(token=HF_TOKEN, model="deepseek-ai/DeepSeek-V3-0324")

def ask_ai(prompt: str) -> str:
    """Synchronous call to DeepSeek via HuggingFace Inference API."""
    try:
        completion = client_hf.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
        )
        return completion.choices[0].message.content
    except Exception as e:
        log.error("HuggingFace API error: %s", e)
        return "Sorry, something went wrong with the AI. 😔"


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_prompt(user_id: str, username: str, user_message: str) -> str:
    """
    Build the full prompt injected into the AI:
      1. Echo persona / personality
      2. Relevant memories from past conversations (vector search)
      3. Current user message
    """
    memories = get_relevant_memories(user_id, user_message)

    parts = [PERSONA]
    if memories:
        parts.append(f"\n{memories}")
    parts.append(f"\n{username}: {user_message}")

    return "\n".join(parts)


# ── Discord bot ───────────────────────────────────────────────────────────────

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix=None, intents=intents)


@bot.event
async def on_ready():
    await bot.tree.sync()
    log.info("Echo bot %s is online!", bot.user)
    await bot.change_presence(activity=discord.Game(name="Echo!"))


@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    await bot.process_commands(message)

    if not bot.user.mentioned_in(message):
        return

    user_id   = str(message.author.id)
    username  = message.author.display_name
    raw_text  = re.sub(r"<@!?[0-9]+>", "", message.content).strip()

    if not raw_text:
        await message.reply("You called? Tell me what's on your mind! 💕")
        return

    async with message.channel.typing():
        # Build prompt with injected memories, run AI in thread pool
        prompt = build_prompt(user_id, username, raw_text)
        reply  = await asyncio.to_thread(ask_ai, prompt)

        # Discord has a 2000-char limit
        if len(reply) > 2000:
            reply = reply[:1990] + "\n\n*(Truncated)*"

        await message.reply(reply)

    # Persist this exchange as a memory in the background
    asyncio.create_task(
        asyncio.to_thread(add_memory, user_id, raw_text, reply)
    )


# ── /draw ─────────────────────────────────────────────────────────────────────

@bot.tree.command(name="draw", description="Free AI image generation with Echo's chaos!")
@app_commands.describe(prompt="What should I draw for you? 🎨")
async def draw(interaction: discord.Interaction, prompt: str):
    await interaction.response.defer()
    await interaction.followup.send(
        f"🎨✨ **DRAWING TIME!!** ✨🎨\n`{prompt}`\n*Give me a sec, creating PURE ART!! ⌚*"
    )

    encoded = prompt.replace(" ", "%20")
    image_url = (
        f"https://image.pollinations.ai/prompt/{encoded}"
        "?width=1024&height=1024&nologo=true&seed=42"
    )

    async with aiohttp.ClientSession() as session:
        async with session.get(image_url) as resp:
            if resp.status != 200:
                await interaction.followup.send("❌ *CRASH* ⚠️ The art engine EXPLODED❗ Try again?")
                return
            data = await resp.read()

    image_file = discord.File(io.BytesIO(data), filename="echo_masterpiece.png")
    embed = discord.Embed(
        title="✨ BEHOLD!! PURE ART!! ✨",
        description=f"**Your prompt:** {prompt}\n\n*Echo approved!! 🎉*",
        color=discord.Color.purple(),
    )
    embed.set_image(url="attachment://echo_masterpiece.png")
    embed.set_footer(text="Made with 💙 by Echo for you! 🎨")

    msg = await interaction.original_response()
    await msg.edit(content=None, embed=embed, attachments=[image_file])


# ── /animate (Pexels stock-video montage) ────────────────────────────────────

@bot.tree.command(
    name="animate",
    description="Generate a short video montage from Pexels stock footage 🎬",
)
@app_commands.describe(prompt="Describe the video you want (e.g. 'ocean waves at sunset')")
async def animate(interaction: discord.Interaction, prompt: str):
    if not PEXELS_API:
        await interaction.response.send_message(
            "❌ `PEXELS_API` is not configured. Ask my creator to add it! 😔",
            ephemeral=True,
        )
        return

    await interaction.response.defer()
    await interaction.followup.send(
        f"🎬 **Finding footage for:** `{prompt}`\n"
        "*Searching Pexels, downloading clips, stitching montage… ⏳ (may take ~30-60 sec)*"
    )

    tmp_dir = tempfile.mkdtemp(prefix="echo_video_")
    try:
        output_path = await asyncio.to_thread(
            create_pexels_video, prompt, PEXELS_API, tmp_dir
        )

        if not output_path or not os.path.exists(output_path):
            await interaction.followup.send(
                f"😔 Couldn't find any matching stock footage for **{prompt}**.\n"
                "Try a more common subject — e.g. *ocean*, *city night*, *forest rain*."
            )
            return

        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        embed = discord.Embed(
            title="🎬 Montage Ready!",
            description=f"**Prompt:** {prompt}\n*Sourced from Pexels stock library 🎥*",
            color=discord.Color.blue(),
        )
        embed.set_footer(text=f"Size: {size_mb:.1f} MB • Made with 💙 by Echo")

        discord_file = discord.File(output_path, filename="echo_montage.mp4")
        await interaction.followup.send(embed=embed, file=discord_file)

    except Exception as e:
        log.exception("animate command failed")
        await interaction.followup.send(f"❌ Something went wrong: `{str(e)[:200]}`")
    finally:
        # Clean up temp directory
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── /memory_status ────────────────────────────────────────────────────────────

@bot.tree.command(
    name="memory_status",
    description="See how many memories Echo has stored about you 🧠",
)
async def memory_status(interaction: discord.Interaction):
    user_id = str(interaction.user.id)
    count   = await asyncio.to_thread(get_memory_count, user_id)
    await interaction.response.send_message(
        f"🧠 I have **{count}** {'memory' if count == 1 else 'memories'} about you, "
        f"{interaction.user.display_name}! 💕\n"
        "Use `/forget_me` if you'd like me to start fresh.",
        ephemeral=True,
    )


# ── /forget_me ────────────────────────────────────────────────────────────────

@bot.tree.command(
    name="forget_me",
    description="Delete all of Echo's memories about you 🗑️",
)
async def forget_me(interaction: discord.Interaction):
    user_id = str(interaction.user.id)
    count   = await asyncio.to_thread(delete_user_memories, user_id)
    if count == 0:
        await interaction.response.send_message(
            "I didn't have any memories about you to delete! 😊",
            ephemeral=True,
        )
    else:
        await interaction.response.send_message(
            f"🗑️ Done! I've wiped **{count}** {'memory' if count == 1 else 'memories'} about you.\n"
            "We're starting completely fresh! 💕",
            ephemeral=True,
        )


# ── Run ───────────────────────────────────────────────────────────────────────

bot.run(DISCORD_TOKEN)
