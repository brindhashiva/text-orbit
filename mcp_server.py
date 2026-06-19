"""
MCP Server for the Multilingual Translator.
Exposes the translate tool so Kiro can call it directly.
"""
import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from translator import translate_text

server = Server("text-orbit")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="translate",
            description=(
                "Translate text into Tamil, English, Kannada, Malayalam, Arabic, "
                "Telugu, Thanglish, Hindi, French, and German."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to translate",
                    },
                    "tone": {
                        "type": "string",
                        "enum": ["formal", "casual"],
                        "description": "Translation tone (formal or casual)",
                        "default": "formal",
                    },
                },
                "required": ["text"],
            },
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name != "translate":
        raise ValueError(f"Unknown tool: {name}")

    text = arguments.get("text", "")
    tone = arguments.get("tone", "formal")

    if not text.strip():
        return [TextContent(type="text", text="Error: text cannot be empty")]

    translations = await translate_text(text, tone)

    output = "\n".join(f"{lang}: {val}" for lang, val in translations.items())
    return [TextContent(type="text", text=output)]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
