import typer
from src.cli.commands import analyze, tools
from src.system.config import settings

app = typer.Typer(
    name="bahis",
    help="Quant Betting Bot – Enterprise Edition",
    add_completion=False,
)

app.add_typer(analyze.app, name="analyze")
app.add_typer(tools.app, name="tools")

if __name__ == "__main__":
    app()
