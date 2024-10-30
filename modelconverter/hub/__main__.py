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

from .enums import License, Order, Task
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

JSONOption: TypeAlias = Annotated[
    bool,
    typer.Option(
        "--json",
        "-j",
        help="Output as JSON.",
        show_default=False,
        is_flag=True,
    ),
]

TeamIDOption: TypeAlias = Annotated[
    Optional[str],
    typer.Option(help="Filter by the team ID", show_default=False),
]
TasksOption: TypeAlias = Annotated[
    Optional[List[Task]],
    typer.Option(
        help="Filter by tasks",
        show_default=False,
    ),
]
UserIDOption: TypeAlias = Annotated[
    Optional[str],
    typer.Option(help="Filter by user ID", show_default=False),
]
SearchOption: TypeAlias = Annotated[
    Optional[str],
    typer.Option(help="Search", show_default=False),
]
LicenseTypeOption: TypeAlias = Annotated[
    Optional[License],
    typer.Option(help="Filter by license type", show_default=False),
]
IsPublicOption: TypeAlias = Annotated[
    bool,
    typer.Option(help="Filter by public models", show_default=False),
]
SlugOption: TypeAlias = Annotated[
    Optional[str],
    typer.Option(help="Filter by slug", show_default=False),
]
ProjectIDOption: TypeAlias = Annotated[
    Optional[str],
    typer.Option(help="Filter by project ID", show_default=False),
]
FilterPublicEntityByTeamIDOption: TypeAlias = Annotated[
    Optional[bool],
    typer.Option(
        help="Whether to filter public entity by team ID", show_default=False
    ),
]
LuxonisOnlyOption: TypeAlias = Annotated[
    bool,
    typer.Option(help="Filter by Luxonis only", show_default=False),
]
CursorOption: TypeAlias = Annotated[
    Optional[str],
    typer.Option(help="Cursor", show_default=False),
]
LimitOption: TypeAlias = Annotated[
    Optional[int],
    typer.Option(help="How many records to display"),
]
SortOption: TypeAlias = Annotated[
    str,
    typer.Option(help="How to sort the results", show_default=False),
]
OrderOption: TypeAlias = Annotated[
    Order,
    typer.Option(help="Order of the sorted results", show_default=False),
]


@app.command()
def download(platform: PlatformOption, slug: SlugArgument):
    params = {"platform": platform.value}
    if slug is not None:
        params["slug"] = slug
    res = Request.get("/api/v1/models/download", params=params)
    print(f"Status code: {res.status_code}")
    print(f"Response: {res.json()}")


def _print_info(model: Dict[str, Any], keys: List[str], **kwargs):
    console = Console()

    if model.get("description_short"):
        description_short_panel = Panel(
            f"[italic]{model['description_short']}[/italic]",
            border_style="dim",
            box=ROUNDED,
        )
    else:
        description_short_panel = None

    table = Table(show_header=False, box=None)
    table.add_column(justify="right", style="bold")
    table.add_column()
    for key in keys:
        if key in ["created", "updated", "last_version_added"]:
            value = model.get(key, "N/A")

            def format_date(date_str):
                try:
                    date_obj = datetime.strptime(
                        date_str, "%Y-%m-%dT%H:%M:%S.%f"
                    )
                    return date_obj.strftime("%B %d, %Y %H:%M:%S")
                except (ValueError, TypeError):
                    return Pretty("N/A")

            formatted_value = format_date(value)

            table.add_row(f"{key.replace('_', ' ').title()}:", formatted_value)
        elif key == "is_public":
            if key not in model:
                value = "N/A"
            else:
                value = "Public" if model[key] else "Private"
            table.add_row("Visibility", Pretty(value))
        elif key == "is_commercial":
            # TODO: Is Usage the right term?
            if key not in model:
                value = "N/A"
            else:
                value = "Commercial" if model[key] else "Non-Commercial"
            table.add_row("Usage:", Pretty(value))
        else:
            table.add_row(
                f"{key.replace('_', ' ').title()}:",
                Pretty(model.get(key, "N/A")),
            )

    info_panel = Panel(
        table, title="Model Information", border_style="cyan", box=ROUNDED
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
def model_info(model_id: str, json: JSONOption = False):
    res = Request.get(f"/api/v1/models/{model_id}")

    if res.status_code != 200:
        print(f"Model with ID '{model_id}' not found.")
        exit(1)

    if json:
        print(res.json())
    else:
        _print_info(
            res.json(),
            keys=[
                "name",
                "slug",
                "id",
                "created",
                "updated",
                "tasks",
                "platforms",
                "is_public",
                "is_commercial",
                "license_type",
                "versions",
                "likes",
                "downloads",
            ],
        )


@version.command(name="info")
def version_info(model_id: str, json: JSONOption = False):
    res = Request.get(
        f"/api/v1/modelVersions/{model_id}",
        params={"team_id": "0192af5f-719d-7b10-8fe6-6d73647dc61a"},
    )
    res = res.json()
    if json:
        print(res)
    else:
        _print_info(
            res,
            keys=[
                "name",
                "slug",
                "version",
                "model_id",
                "id",
                "created",
                "updated",
                "platforms",
                "exportable_to",
                "is_public",
            ],
        )


@instance.command(name="info")
def instance_info(model_id: str, json: JSONOption = False):
    res = Request.get(f"/api/v1/modelInstances/{model_id}").json()
    if json:
        print(res)
    else:
        _print_info(
            res.json(),
            keys=[
                "name",
                "slug",
                "id",
                "last_version_added",
                "platforms",
                "is_public",
                "is_commercial",
                "license_type",
                "versions",
                "likes",
                "downloads",
            ],
        )


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


def _ls(
    endpoint: str,
    team_id: Optional[str],
    tasks: Optional[List[Task]] = None,
    user_id: Optional[str] = None,
    search: Optional[str] = None,
    license_type: Optional[License] = None,
    is_public: bool = True,
    slug: Optional[str] = None,
    project_id: Optional[str] = None,
    filter_public_entity_by_team_id: Optional[bool] = None,
    luxonis_only: bool = False,
    cursor: Optional[str] = None,
    limit: Optional[int] = 50,
    sort: str = "updated",
    order: Order = Order.DESC,
    **kwargs,
) -> None:
    params = {
        "team_id": team_id,
        "tasks": [task.name for task in tasks] if tasks else None,
        "user_id": user_id,
        "search": search,
        "license_type": license_type,
        "is_public": is_public,
        "slug": slug,
        "project_id": project_id,
        "filter_public_entity_by_team_id": filter_public_entity_by_team_id,
        "luxonis_only": luxonis_only,
        "cursor": cursor,
        "limit": limit,
        "sort": sort,
        "order": order,
    }
    res = Request.get(
        f"/api/v1/{endpoint}/{team_id or ''}", params=params
    ).json()
    console = Console()
    console.print(
        _get_table(
            res,
            **kwargs,
            row_styles=["yellow", "cyan"],
            box=rich.box.ROUNDED,
        ),
    )


@model.command(name="ls")
def model_ls(
    team_id: TeamIDOption = None,
    tasks: TasksOption = None,
    user_id: UserIDOption = None,
    search: SearchOption = None,
    license_type: LicenseTypeOption = None,
    is_public: IsPublicOption = True,
    slug: SlugOption = None,
    project_id: ProjectIDOption = None,
    filter_public_entity_by_team_id: FilterPublicEntityByTeamIDOption = None,
    luxonis_only: LuxonisOnlyOption = False,
    cursor: CursorOption = None,
    limit: LimitOption = 50,
    sort: SortOption = "updated",
    order: OrderOption = Order.DESC,
):
    _ls(
        "models",
        team_id=team_id,
        tasks=tasks,
        user_id=user_id,
        search=search,
        license_type=license_type,
        is_public=is_public,
        slug=slug,
        project_id=project_id,
        filter_public_entity_by_team_id=filter_public_entity_by_team_id,
        luxonis_only=luxonis_only,
        cursor=cursor,
        limit=limit,
        sort=sort,
        order=order,
        keys=["name", "id", "slug"],
    )


@version.command(name="ls")
def version_ls(
    team_id: TeamIDOption = None,
    tasks: TasksOption = None,
    user_id: UserIDOption = None,
    search: SearchOption = None,
    license_type: LicenseTypeOption = None,
    is_public: IsPublicOption = True,
    slug: SlugOption = None,
    project_id: ProjectIDOption = None,
    filter_public_entity_by_team_id: FilterPublicEntityByTeamIDOption = None,
    luxonis_only: LuxonisOnlyOption = False,
    cursor: CursorOption = None,
    limit: LimitOption = 50,
    sort: SortOption = "updated",
    order: OrderOption = Order.DESC,
):
    _ls(
        "modelVersions",
        team_id=team_id,
        tasks=tasks,
        user_id=user_id,
        search=search,
        license_type=license_type,
        is_public=is_public,
        slug=slug,
        project_id=project_id,
        filter_public_entity_by_team_id=filter_public_entity_by_team_id,
        luxonis_only=luxonis_only,
        cursor=cursor,
        limit=limit,
        sort=sort,
        order=order,
        keys=["id", "model_id", "version", "slug", "platforms"],
    )


@instance.command(name="ls")
def instance_ls(
    team_id: TeamIDOption = None,
    tasks: TasksOption = None,
    user_id: UserIDOption = None,
    search: SearchOption = None,
    license_type: LicenseTypeOption = None,
    is_public: IsPublicOption = True,
    slug: SlugOption = None,
    project_id: ProjectIDOption = None,
    filter_public_entity_by_team_id: FilterPublicEntityByTeamIDOption = None,
    luxonis_only: LuxonisOnlyOption = False,
    cursor: CursorOption = None,
    limit: LimitOption = 50,
    sort: SortOption = "updated",
    order: OrderOption = Order.DESC,
):
    _ls(
        "modelInstances",
        team_id=team_id,
        tasks=tasks,
        user_id=user_id,
        search=search,
        license_type=license_type,
        is_public=is_public,
        slug=slug,
        project_id=project_id,
        filter_public_entity_by_team_id=filter_public_entity_by_team_id,
        luxonis_only=luxonis_only,
        cursor=cursor,
        limit=limit,
        sort=sort,
        order=order,
        keys=["id", "instance_id", "slug", "platforms"],
    )


@app.command()
def convert():
    logger.info("Converting model")
