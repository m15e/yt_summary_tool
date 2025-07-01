#!/usr/bin/env python3
import re
import os
import argparse
import pyperclip
from youtube_transcript_api import YouTubeTranscriptApi
from openai import OpenAI
import google.generativeai as genai
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.text import Text

# Load environment variables from .env file
load_dotenv()

# Initialize Rich console
console = Console()


def summary_prompt(text, question=None):
    base_prompt = f"""**Objective:**

Summarize the following text succinctly while preserving the key points, main ideas, and essential details. Ensure clarity and conciseness in your summary.

**Text:**

{text}

**Your Summary:**

Provide a structured and concise summary based on the guidelines above.
Use bulleted lists to break down the main points."""
    
    if question:
        base_prompt += f"""

**Additional Question:**

{question}

**Question Answer:**

Also answer the above question based on the text provided."""
    
    base_prompt += "\n\nReturn your response in markdown."
    return base_prompt


def get_video_id(url):
    """Extract video ID from YouTube URL"""
    patterns = [r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", r"(?:embed\/)([0-9A-Za-z_-]{11})"]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def get_youtube_transcript(video_id):
    """Get transcript from YouTube"""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([entry["text"] for entry in transcript])
    except Exception as e:
        console.print(f"[red]Error getting transcript: {e}[/red]")
        return None


def transcribe(url):
    video_id = get_video_id(url)
    if not video_id:
        console.print("[red]Error: Could not extract video ID from URL[/red]")
        return None
    return get_youtube_transcript(video_id)


def llm_summary(text, llm="Gemini", question=None):
    if not text:
        console.print("[red]Error: No text provided for summarization[/red]")
        return None

    prompt = summary_prompt(text, question)

    if llm == "OpenAI":
        # if not os.environ.get("OPENAI"):
        #     print("Error: OPENAI environment variable not set")
        #     return None
        client = OpenAI(api_key=os.environ.get("OPENAI"))
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that summarizes YouTube video transcripts.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            console.print(f"[red]Error with OpenAI API: {e}[/red]")
            return None

    elif llm == "Gemini":
        if not os.environ.get("GMNI"):
            console.print("[red]Error: GMNI environment variable not set[/red]")
            return None
        try:
            genai.configure(api_key=os.environ.get("GMNI"))
            model = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            console.print(f"[red]Error with Gemini API: {e}[/red]")
            return None
    else:
        console.print("[red]Error: Unsupported LLM. Choose either 'OpenAI' or 'Gemini'[/red]")
        return None


def main():
    parser = argparse.ArgumentParser(description="Summarize YouTube videos using AI")
    parser.add_argument("url", nargs='?', help="YouTube video URL (optional - will use clipboard if not provided)")
    parser.add_argument(
        "--llm",
        default="OpenAI",
        choices=["OpenAI", "Gemini"],
        help="LLM to use for summarization (default: OpenAI)",
    )
    parser.add_argument(
        "-q", "--question",
        help="Optional question to ask about the transcript",
    )

    args = parser.parse_args()
    
    # Use clipboard URL if no URL provided as argument
    if not args.url:
        try:
            clipboard_content = pyperclip.paste().strip()
            if clipboard_content and ('youtube.com' in clipboard_content or 'youtu.be' in clipboard_content):
                args.url = clipboard_content
                console.print(f"[yellow]📋 Using URL from clipboard: {args.url}[/yellow]")
            else:
                console.print("[red]Error: No YouTube URL provided and clipboard doesn't contain a YouTube URL[/red]")
                parser.print_help()
                return
        except Exception as e:
            console.print(f"[red]Error accessing clipboard: {e}[/red]")
            parser.print_help()
            return

    console.print(Panel(f"[bold blue]Processing YouTube video:[/bold blue] {args.url}", title="🎬 YouTube Summarizer"))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task1 = progress.add_task("🔍 Extracting transcript...", total=None)
        transcript = transcribe(args.url)
        progress.remove_task(task1)

        if not transcript:
            console.print(
                Panel(
                    "[red]Failed to get transcript. The video might not have captions available.[/red]",
                    title="❌ Error",
                    border_style="red"
                )
            )
            return

        task_desc = f"🧠 Generating summary using {args.llm}..."
        if args.question:
            task_desc = f"🧠 Generating summary and answering question using {args.llm}..."
        task2 = progress.add_task(task_desc, total=None)
        summary = llm_summary(transcript, args.llm, args.question)
        progress.remove_task(task2)

    if summary:
        pyperclip.copy(summary)
        console.print("[green]📋 Summary copied to clipboard![/green]")
        console.print()
        
        # Render the summary as markdown in a panel
        markdown = Markdown(summary)
        console.print(
            Panel(
                markdown,
                title="📝 Summary",
                border_style="green",
                padding=(1, 2)
            )
        )
    else:
        console.print(
            Panel(
                "[red]Failed to generate summary.[/red]",
                title="❌ Error",
                border_style="red"
            )
        )


if __name__ == "__main__":
    main()
