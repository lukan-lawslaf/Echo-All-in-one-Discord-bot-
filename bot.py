import os
import os
import discord
from discord import app_commands
from discord.ext import commands
from google import genai
from google.genai import types
from dotenv import load_dotenv
import re
from google.genai.types import HttpOptions
import io
import aiohttp
from gradio_client import Client
import urllib.parse
from huggingface_hub import login
import asyncio
from gradio_client import Client, handle_file
import requests  # ADD THIS LINE
import sys
from huggingface_hub import InferenceClient
# Ensure UTF-8 Encoding
sys.stdout.reconfigure(encoding="utf-8")

with open("chat.txt", "r+", encoding="utf-8") as file: # Load chat context accordingly (chat4.txt, chat3.txt, etc.)
    chat = file.read()
    chat += "\nUser: "

# Load variables from .env
load_dotenv()
DISCORD_TOKEN = os.getenv("Discord_Token")
GEMINI_KEY = os.getenv("G_Api")
HF_TOKEN = os.getenv("HF_TOKEN")

# Initialize Hugging Face client with DeepSeek
client_hf = InferenceClient(token=HF_TOKEN, model="deepseek-ai/DeepSeek-V3-0324")

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix=None, intents=intents)

def ask_ai(prompt):
    try:
        completion = client_hf.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"!!! HF ERROR: {e}") 
        return "Sorry, something went wrong with the AI."
@bot.event
async def on_ready():
    await bot.tree.sync()
    # Removed emoji to avoid the Windows terminal crash you had earlier
    print(f"Echo Bot {bot.user} is online! Mention me to chat.")
    await bot.change_presence(activity=discord.Game(name="Echo!"))

@bot.event
async def on_message(message):
    if message.author.bot:
        return

    if bot.user.mentioned_in(message):
        # IMPROVED CLEANING: Uses Regex to remove ANY mention format cleanly
        clean_text = chat+re.sub(r'<@!?[0-9]+>', '', message.content).strip()
        
        if not clean_text:
            await message.reply("You called? Tell me what's on your mind!")
            return

        async with message.channel.typing():
            reply = ask_ai(clean_text)
            
            # Check if it's too long for Discord
            if len(reply) > 2000:
                # Cut it to 1990 chars to leave room for the note
                safe_text = reply[:1990] + "\n\n*(Truncated due to length)*"
            else:
                safe_text = reply

            await message.reply(safe_text)
            
@bot.tree.command(name="draw", description="Free AI image generation with Echo's chaos!")
async def draw(interaction: discord.Interaction, prompt: str):
    await interaction.response.defer()
    await interaction.followup.send(f"🎨✨ **DRAWING TIME!!** ✨🎨\n`{prompt}`\n* Give me a sec, creating PURE ART!! ⌚*")

    encoded_prompt = prompt.replace(' ', '%20')
    image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&nologo=true&seed=42"

    async with aiohttp.ClientSession() as session:
        async with session.get(image_url) as resp:
            if resp.status != 200:
                await interaction.followup.send("❌ *CRASH* ⚠️ The art engine EXPLODED❗ Try again❓")
                return
            data = await resp.read()

    image_file = discord.File(io.BytesIO(data), filename="echo_masterpiece.png")

    embed = discord.Embed(
        title="✨ BEHOLD!! PURE ART!! ✨", 
        description=f"**Your prompt:** {prompt}\n\n*Echo approved!! 🎉*", 
        color=discord.Color.purple()
    )
    embed.set_image(url="attachment://echo_masterpiece.png")
    embed.set_footer(text=" Made with 💙 by Echo for you!🎨")

    message = await interaction.original_response()
    await message.edit(content=None, embed=embed, attachments=[image_file])


# Add this at the VERY TOP of your file with other imports
import subprocess 

# --- IMPORTS (Top of file) ---
from kaggle.api.kaggle_api_extended import KaggleApi
import asyncio

bot.run(DISCORD_TOKEN)