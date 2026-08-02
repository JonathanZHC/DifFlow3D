#!/usr/bin/env python3
"""Create and animate the three objects used by the dynamic scene mode.

The module owns all dynamic geometry and trajectories. ``scene_settings.py``
only creates the common ground, table and lighting.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from pxr import Gf, Sdf, UsdGeom, UsdShade


@dataclass
class _MovingObject:
    name: str
    translate_op: Any
    rotate_op: Any


def _define_xform(stage, path: str):
    return UsdGeom.Xform.Define(stage, path)


def _create_material(
    stage,
    path: str,
    color: tuple[float, float, float],
    roughness: float = 0.35,
    metallic: float = 0.0,
) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput(
        "diffuseColor",
        Sdf.ValueTypeNames.Color3f,
    ).Set(Gf.Vec3f(*color))
    shader.CreateInput(
        "roughness",
        Sdf.ValueTypeNames.Float,
    ).Set(float(roughness))
    shader.CreateInput(
        "metallic",
        Sdf.ValueTypeNames.Float,
    ).Set(float(metallic))
    material.CreateSurfaceOutput().ConnectToSource(
        shader.ConnectableAPI(),
        "surface",
    )
    return material


def _bind_material(prim, material: UsdShade.Material) -> None:
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def _make_motion_root(
    stage,
    path: str,
) -> tuple[Any, Any]:
    root = _define_xform(stage, path)
    xformable = UsdGeom.Xformable(root.GetPrim())
    xformable.ClearXformOpOrder()
    translate_op = xformable.AddTranslateOp()
    rotate_op = xformable.AddRotateXYZOp()
    return translate_op, rotate_op


def _set_local_transform(
    prim,
    translation: tuple[float, float, float],
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    rotation_xyz_deg: tuple[float, float, float] | None = None,
) -> None:
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(Gf.Vec3d(*translation))
    if rotation_xyz_deg is not None:
        xformable.AddRotateXYZOp().Set(
            Gf.Vec3f(*rotation_xyz_deg)
        )
    xformable.AddScaleOp().Set(Gf.Vec3f(*scale))


def _create_local_cube(
    stage,
    path: str,
    center: tuple[float, float, float],
    size_xyz: tuple[float, float, float],
    material: UsdShade.Material,
    rotation_xyz_deg: tuple[float, float, float] | None = None,
) -> None:
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    _set_local_transform(
        cube.GetPrim(),
        center,
        size_xyz,
        rotation_xyz_deg,
    )
    _bind_material(cube.GetPrim(), material)


def _create_local_sphere(
    stage,
    path: str,
    center: tuple[float, float, float],
    radius: float,
    material: UsdShade.Material,
    scale_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> None:
    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(float(radius))
    _set_local_transform(
        sphere.GetPrim(),
        center,
        scale_xyz,
    )
    _bind_material(sphere.GetPrim(), material)


def _create_local_cylinder(
    stage,
    path: str,
    center: tuple[float, float, float],
    radius: float,
    height: float,
    material: UsdShade.Material,
    rotation_xyz_deg: tuple[float, float, float] | None = None,
) -> None:
    cylinder = UsdGeom.Cylinder.Define(stage, path)
    cylinder.CreateAxisAttr(UsdGeom.Tokens.z)
    cylinder.CreateRadiusAttr(float(radius))
    cylinder.CreateHeightAttr(float(height))
    _set_local_transform(
        cylinder.GetPrim(),
        center,
        (1.0, 1.0, 1.0),
        rotation_xyz_deg,
    )
    _bind_material(cylinder.GetPrim(), material)


def _build_tall_shelf(
    stage,
    root_path: str,
    materials: dict[str, UsdShade.Material],
) -> None:
    # Root origin is slightly above the table surface, at wheel-centre height.
    for index, z in enumerate((0.07, 0.30, 0.53)):
        _create_local_cube(
            stage,
            f"{root_path}/Shelf_{index}",
            (0.0, 0.0, z),
            (0.34, 0.24, 0.035),
            materials["metal"],
        )

    for index, (x, y) in enumerate(
        (
            (-0.15, -0.10),
            (-0.15, 0.10),
            (0.15, -0.10),
            (0.15, 0.10),
        )
    ):
        _create_local_cube(
            stage,
            f"{root_path}/Post_{index}",
            (x, y, 0.30),
            (0.028, 0.028, 0.58),
            materials["dark"],
        )

    # Cross braces add non-trivial slanted surfaces.
    _create_local_cube(
        stage,
        f"{root_path}/Brace_A",
        (0.0, 0.115, 0.30),
        (0.026, 0.022, 0.53),
        materials["blue"],
        (0.0, 31.0, 0.0),
    )
    _create_local_cube(
        stage,
        f"{root_path}/Brace_B",
        (0.0, 0.115, 0.30),
        (0.026, 0.022, 0.53),
        materials["blue"],
        (0.0, -31.0, 0.0),
    )

    _create_local_cube(
        stage,
        f"{root_path}/UpperBox",
        (-0.06, -0.01, 0.615),
        (0.13, 0.13, 0.13),
        materials["yellow"],
        (0.0, 0.0, 7.0),
    )
    _create_local_cylinder(
        stage,
        f"{root_path}/MiddleCan",
        (0.07, 0.0, 0.385),
        0.042,
        0.11,
        materials["red"],
    )

    for index, (x, y) in enumerate(
        (
            (-0.13, -0.115),
            (-0.13, 0.115),
            (0.13, -0.115),
            (0.13, 0.115),
        )
    ):
        _create_local_cylinder(
            stage,
            f"{root_path}/Wheel_{index}",
            (x, y, 0.0),
            0.035,
            0.025,
            materials["dark"],
            (90.0, 0.0, 0.0),
        )


def _build_tall_bottle(
    stage,
    root_path: str,
    materials: dict[str, UsdShade.Material],
) -> None:
    _create_local_cylinder(
        stage,
        f"{root_path}/Body",
        (0.0, 0.0, 0.145),
        0.063,
        0.29,
        materials["green"],
    )
    _create_local_cylinder(
        stage,
        f"{root_path}/BottomRing",
        (0.0, 0.0, 0.025),
        0.067,
        0.03,
        materials["dark"],
    )
    _create_local_cylinder(
        stage,
        f"{root_path}/Label",
        (0.0, 0.0, 0.16),
        0.065,
        0.095,
        materials["white"],
    )
    _create_local_sphere(
        stage,
        f"{root_path}/Shoulder",
        (0.0, 0.0, 0.29),
        0.064,
        materials["green"],
        (1.0, 1.0, 0.58),
    )
    _create_local_cylinder(
        stage,
        f"{root_path}/Neck",
        (0.0, 0.0, 0.35),
        0.028,
        0.115,
        materials["green"],
    )
    _create_local_cylinder(
        stage,
        f"{root_path}/NeckRing",
        (0.0, 0.0, 0.392),
        0.034,
        0.022,
        materials["metal"],
    )
    _create_local_cylinder(
        stage,
        f"{root_path}/Cap",
        (0.0, 0.0, 0.425),
        0.034,
        0.048,
        materials["red"],
    )


def _build_floating_object(
    stage,
    root_path: str,
    materials: dict[str, UsdShade.Material],
) -> None:
    _create_local_sphere(
        stage,
        f"{root_path}/Core",
        (0.0, 0.0, 0.0),
        0.085,
        materials["blue"],
        (1.30, 0.95, 0.58),
    )
    _create_local_cylinder(
        stage,
        f"{root_path}/TopDome",
        (0.0, 0.0, 0.052),
        0.052,
        0.058,
        materials["white"],
    )

    _create_local_cube(
        stage,
        f"{root_path}/ArmFront",
        (0.145, 0.0, 0.0),
        (0.21, 0.028, 0.022),
        materials["metal"],
    )
    _create_local_cube(
        stage,
        f"{root_path}/ArmRear",
        (-0.145, 0.0, 0.0),
        (0.21, 0.028, 0.022),
        materials["metal"],
    )
    _create_local_cube(
        stage,
        f"{root_path}/ArmLeft",
        (0.0, 0.145, 0.0),
        (0.028, 0.21, 0.022),
        materials["metal"],
    )
    _create_local_cube(
        stage,
        f"{root_path}/ArmRight",
        (0.0, -0.145, 0.0),
        (0.028, 0.21, 0.022),
        materials["metal"],
    )

    for name, center in (
        ("Front", (0.25, 0.0, 0.0)),
        ("Rear", (-0.25, 0.0, 0.0)),
        ("Left", (0.0, 0.25, 0.0)),
        ("Right", (0.0, -0.25, 0.0)),
    ):
        _create_local_sphere(
            stage,
            f"{root_path}/Pod{name}",
            center,
            0.060,
            materials["yellow"],
            (1.0, 1.0, 0.48),
        )
        _create_local_cylinder(
            stage,
            f"{root_path}/Rotor{name}",
            (center[0], center[1], 0.043),
            0.085,
            0.010,
            materials["dark"],
        )


class MovingObjectController:
    """Create and update all three scripted compound obstacles."""

    SHELF_PERIOD_S = 11.0
    BOTTLE_PERIOD_S = 8.5
    FLOATING_PERIOD_S = 10.0

    def __init__(
        self,
        stage,
        table_surface_z: float,
        speed_scale: float = 1.0,
    ) -> None:
        if speed_scale <= 0.0:
            raise ValueError("speed_scale must be positive.")

        self.speed_scale = float(speed_scale)
        self.table_surface_z = float(table_surface_z)
        self.objects: dict[str, _MovingObject] = {}

        _define_xform(stage, "/World/MovingObjects")
        _define_xform(stage, "/World/Materials")

        materials = {
            "metal": _create_material(
                stage,
                "/World/Materials/MovingMetal",
                (0.48, 0.52, 0.58),
                roughness=0.22,
                metallic=0.72,
            ),
            "dark": _create_material(
                stage,
                "/World/Materials/MovingDark",
                (0.045, 0.055, 0.065),
                roughness=0.52,
            ),
            "blue": _create_material(
                stage,
                "/World/Materials/MovingBlue",
                (0.035, 0.20, 0.78),
                roughness=0.28,
            ),
            "yellow": _create_material(
                stage,
                "/World/Materials/MovingYellow",
                (0.95, 0.58, 0.025),
                roughness=0.34,
            ),
            "red": _create_material(
                stage,
                "/World/Materials/MovingRed",
                (0.78, 0.035, 0.025),
                roughness=0.30,
            ),
            "green": _create_material(
                stage,
                "/World/Materials/MovingGreen",
                (0.025, 0.52, 0.14),
                roughness=0.30,
            ),
            "white": _create_material(
                stage,
                "/World/Materials/MovingWhite",
                (0.86, 0.88, 0.93),
                roughness=0.38,
            ),
        }

        shelf_path = "/World/MovingObjects/TallShelf"
        shelf_translate, shelf_rotate = _make_motion_root(
            stage,
            shelf_path,
        )
        _build_tall_shelf(stage, shelf_path, materials)
        self.objects["tall_shelf"] = _MovingObject(
            name="tall_shelf",
            translate_op=shelf_translate,
            rotate_op=shelf_rotate,
        )

        bottle_path = "/World/MovingObjects/TallBottle"
        bottle_translate, bottle_rotate = _make_motion_root(
            stage,
            bottle_path,
        )
        _build_tall_bottle(stage, bottle_path, materials)
        self.objects["tall_bottle"] = _MovingObject(
            name="tall_bottle",
            translate_op=bottle_translate,
            rotate_op=bottle_rotate,
        )

        floating_path = "/World/MovingObjects/FloatingObject"
        floating_translate, floating_rotate = _make_motion_root(
            stage,
            floating_path,
        )
        _build_floating_object(stage, floating_path, materials)
        self.objects["floating_object"] = _MovingObject(
            name="floating_object",
            translate_op=floating_translate,
            rotate_op=floating_rotate,
        )

        self.update(0.0)

    @staticmethod
    def _phase(
        elapsed_s: float,
        period_s: float,
    ) -> float:
        return 2.0 * math.pi * elapsed_s / period_s

    def update(self, elapsed_s: float) -> None:
        """Update all enabled object transforms."""

        t = max(0.0, float(elapsed_s)) * self.speed_scale

        shelf = self.objects.get("tall_shelf")
        if shelf is not None:
            phase = self._phase(t, self.SHELF_PERIOD_S)
            x = -0.30 + 0.18 * math.sin(phase)
            y = 0.25 + 0.055 * math.sin(0.55 * phase + 0.8)
            # Wheel centres are 35 mm above the tabletop.
            z = self.table_surface_z + 0.035
            yaw_deg = 9.0 * math.sin(0.72 * phase)

            shelf.translate_op.Set(Gf.Vec3d(x, y, z))
            shelf.rotate_op.Set(
                Gf.Vec3f(0.0, 0.0, yaw_deg)
            )

        bottle = self.objects.get("tall_bottle")
        if bottle is not None:
            phase = self._phase(t, self.BOTTLE_PERIOD_S)
            x = 0.30 + 0.12 * math.cos(phase)
            y = 0.12 + 0.10 * math.sin(phase)
            z = self.table_surface_z

            bottle.translate_op.Set(Gf.Vec3d(x, y, z))
            bottle.rotate_op.Set(
                Gf.Vec3f(
                    3.5 * math.sin(1.4 * phase),
                    2.0 * math.sin(0.8 * phase + 0.4),
                    22.0 * math.sin(phase),
                )
            )

        floating = self.objects.get("floating_object")
        if floating is not None:
            phase = self._phase(t, self.FLOATING_PERIOD_S)
            x = 0.02 + 0.25 * math.sin(phase)
            y = -0.23 + 0.13 * math.sin(1.55 * phase + 1.0)
            z = (
                self.table_surface_z
                + 0.47
                + 0.10 * math.sin(2.1 * phase)
            )

            floating.translate_op.Set(Gf.Vec3d(x, y, z))
            floating.rotate_op.Set(
                Gf.Vec3f(
                    6.0 * math.sin(1.3 * phase),
                    8.0 * math.sin(0.9 * phase + 0.3),
                    math.degrees(0.75 * phase) % 360.0,
                )
            )

    def description(self) -> str:
        enabled = ", ".join(self.objects.keys())
        return enabled if enabled else "none"
