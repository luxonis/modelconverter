import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests
import rich.box
import typer
from rich import print
from rich.box import ROUNDED
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.pretty import Pretty
from rich.table import Table
from typing_extensions import Annotated, TypeAlias

from modelconverter.utils.types import Target

from .hub_requests import Request

logger = logging.getLogger(__name__)
app = typer.Typer(
    help="Hub CLI",
    add_completion=False,
    rich_markup_mode="markdown",
)

model = typer.Typer(
    help="Models Interactions",
    add_completion=False,
    rich_markup_mode="markdown",
)

version = typer.Typer(
    help="Hub Versions Interactions",
    add_completion=False,
    rich_markup_mode="markdown",
)

instance = typer.Typer(
    help="Hub Instances Interactions",
    add_completion=False,
    rich_markup_mode="markdown",
)

app.add_typer(
    model,
    name="model",
    help="Models Interactions",
    invoke_without_command=True,
)
app.add_typer(version, name="version", help="Hub Versions Interactions")
app.add_typer(instance, name="instance", help="Hub Instances Interactions")


def print_response(res: requests.Response):
    table = Table(
        title="Response",
        box=rich.box.ROUNDED,
        width=74,
    )
    table.add_column("Status Code", header_style="magenta i")
    table.add_column("JSON", header_style="magenta i")
    table.add_row(Pretty(res.status_code), Pretty(res.json()))
    print(table)


PlatformOption: TypeAlias = Annotated[
    Target,
    typer.Option(
        case_sensitive=False,
        help="What platform to convert the model to.",
        show_default=False,
    ),
]

SlugArgument: TypeAlias = Annotated[
    Optional[str],
    typer.Argument(
        show_default=False,
        help="The model slug.",
    ),
]


@app.command()
def download(platform: PlatformOption, slug: SlugArgument):
    params = {"platform": platform.value}
    if slug is not None:
        params["slug"] = slug
    res = Request.get("/api/v1/models/download", params=params)
    print(f"Status code: {res.status_code}")
    print(f"Response: {res.json()}")


def print_model_info(model: Dict[str, Any]):
    console = Console()

    if model.get("description_short"):
        description_short_panel = Panel(
            f"[italic]{model['description_short']}[/italic]",
            border_style="dim",
            box=ROUNDED,
        )
    else:
        description_short_panel = None

    info_table = Table(show_header=False, box=None)
    info_table.add_column(justify="right", style="bold")
    info_table.add_column()

    info_table.add_row("Name:", model.get("name", "N/A"))
    info_table.add_row("Slug:", model.get("slug", "N/A"))
    info_table.add_row("Versions:", str(model.get("versions", 0)))
    created_str = model.get("created", "N/A")
    updated_str = model.get("updated", "N/A")

    def format_date(date_str):
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%f")
            return date_obj.strftime("%B %d, %Y %H:%M:%S")
        except (ValueError, TypeError):
            return "N/A"

    created_formatted = format_date(created_str)
    updated_formatted = format_date(updated_str)

    info_table.add_row("Created:", created_formatted)
    info_table.add_row("Updated:", updated_formatted)
    info_table.add_row("Tasks:", ", ".join(model.get("tasks", [])) or "N/A")
    platforms = ", ".join(model.get("platforms", [])) or "N/A"
    info_table.add_row("Platforms:", platforms)
    visibility = "Public" if model.get("is_public") else "Private"
    usage = "Commercial" if model.get("is_commercial") else "Non-Commercial"
    info_table.add_row("Visibility:", visibility)
    info_table.add_row("Usage:", usage)
    info_table.add_row("License:", model.get("license_type", "Unknown"))
    info_table.add_row("Likes:", str(model.get("likes", 0)))
    info_table.add_row("Downloads:", str(model.get("downloads", 0)))

    info_panel = Panel(
        info_table, title="Model Information", border_style="cyan", box=ROUNDED
    )

    if model.get("description"):
        description_panel = Panel(
            Markdown(model["description"]),
            title="Description",
            border_style="green",
            box=ROUNDED,
        )
    else:
        description_panel = None

    nested_panels = []
    if description_short_panel:
        nested_panels.append(description_short_panel)
    nested_panels.append(info_panel)
    if description_panel:
        nested_panels.append(description_panel)

    content = Group(*nested_panels)

    main_panel = Panel(
        content,
        title=f"[bold magenta]{model.get('name', 'N/A')}[/bold magenta]",
        width=74,
        border_style="magenta",
        box=ROUNDED,
    )

    console.print(main_panel)
    console.rule()


@model.command(name="info")
def model_info(model_id: str):
    res = Request.get(f"/api/v1/models/{model_id}")
    print_model_info(res.json())


@model.command(name="create")
def model_create(
    name: str,
    license_type: Optional[str] = None,
    public: bool = True,
    description_short: Optional[str] = None,
    description: Optional[str] = None,
    architecture_id: Optional[str] = None,
):
    data = {
        "name": name,
        "license_type": license_type,
        "is_public": public,
        "description_short": description_short,
    }
    if description_short is not None:
        data["description_short"] = description_short
    if description is not None:
        data["description"] = description
    if architecture_id is not None:
        data["architecture_id"] = architecture_id
    res = Request.post("/api/v1/models", json=data)
    print_response(res)


@model.command(name="delete")
def model_delete(model_id: str):
    res = Request.delete(f"/api/v1/models/{model_id}")
    print_response(res)


def _get_table(data: List[Dict[str, Any]], keys: List[str], **kwargs) -> Table:
    table = Table(**kwargs)
    for key in keys:
        table.add_column(key, header_style="magenta i")

    for model in data:
        renderables = []
        for key in keys:
            value = model.get(key, "N/A")
            if isinstance(value, list):
                value = ", ".join(value)
            renderables.append(value)
        table.add_row(*renderables)

    return table


@model.command(name="ls")
def model_ls(team_id: Optional[str] = None, is_public: bool = True):
    data = {
        "is_public": is_public,
    }
    res = Request.get(f"/api/v1/models/{team_id or ''}", params=data).json()
    print(
        _get_table(
            res,
            ["name", "team_id", "slug"],
            title="Models",
            box=rich.box.ROUNDED,
            row_styles=["yellow", "cyan"],
            width=74,
        )
    )


@version.command(name="ls")
def version_ls(team_id: Optional[str] = None, is_public: bool = True):
    data = {
        "is_public": is_public,
    }
    res = Request.get(
        f"/api/v1/modelVersions/{team_id or ''}", params=data
    ).json()
    print(
        _get_table(
            res,
            ["model_id", "version", "slug", "platforms"],
            title="Versions",
            box=rich.box.ROUNDED,
            row_styles=["yellow", "cyan"],
            width=74,
        )
    )


@app.command()
def convert():
    logger.info("Converting model")
