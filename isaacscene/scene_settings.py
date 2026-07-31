#!/usr/bin/env python3
"""程序化创建桌面日常物体场景。

本模块不创建 SimulationApp，也不创建相机或 ROS 节点。
必须在 run_isaacsim.py 创建 SimulationApp 之后导入。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade


@dataclass(frozen=True)
class SceneConfig:
    """桌面场景几何参数，单位均为米。"""

    table_top_center_z: float = 0.75
    table_top_size_x: float = 1.60
    table_top_size_y: float = 1.15
    table_top_thickness: float = 0.08
    ground_size: float = 5.0


DEFAULT_SCENE_CONFIG = SceneConfig()


def _define_xform(stage, path: str) -> None:
    UsdGeom.Xform.Define(stage, path)


def _set_xform(
    prim,
    translation: tuple[float, float, float],
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    rotation_xyz_deg: tuple[float, float, float] | None = None,
) -> None:
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(Gf.Vec3d(*translation))
    if rotation_xyz_deg is not None:
        xformable.AddRotateXYZOp().Set(Gf.Vec3f(*rotation_xyz_deg))
    xformable.AddScaleOp().Set(Gf.Vec3f(*scale))


def _create_material(
    stage,
    path: str,
    color: tuple[float, float, float],
    roughness: float,
    metallic: float = 0.0,
) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*color)
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(
        float(roughness)
    )
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(
        float(metallic)
    )
    material.CreateSurfaceOutput().ConnectToSource(
        shader.ConnectableAPI(), "surface"
    )
    return material


def _bind_material(prim, material: UsdShade.Material) -> None:
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def _create_cube(
    stage,
    path: str,
    center: tuple[float, float, float],
    size_xyz: tuple[float, float, float],
    material: UsdShade.Material,
    rotation_xyz_deg: tuple[float, float, float] | None = None,
) -> None:
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    _set_xform(cube.GetPrim(), center, size_xyz, rotation_xyz_deg)
    _bind_material(cube.GetPrim(), material)


def _create_cylinder(
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
    _set_xform(
        cylinder.GetPrim(),
        center,
        (1.0, 1.0, 1.0),
        rotation_xyz_deg,
    )
    _bind_material(cylinder.GetPrim(), material)


def _create_sphere(
    stage,
    path: str,
    center: tuple[float, float, float],
    radius: float,
    material: UsdShade.Material,
    scale_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> None:
    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(float(radius))
    _set_xform(sphere.GetPrim(), center, scale_xyz)
    _bind_material(sphere.GetPrim(), material)


def build_scene(
    stage,
    config: SceneConfig = DEFAULT_SCENE_CONFIG,
) -> Dict[str, str]:
    """创建场景并返回逻辑对象名到 USD prim 路径的映射。"""

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    for path in (
        "/World",
        "/World/Table",
        "/World/Objects",
        "/World/Cameras",
        "/World/Lights",
        "/World/Materials",
    ):
        _define_xform(stage, path)

    materials = {
        "ground": _create_material(
            stage, "/World/Materials/Ground", (0.18, 0.20, 0.23), 0.85
        ),
        "wood": _create_material(
            stage, "/World/Materials/Wood", (0.42, 0.20, 0.07), 0.58
        ),
        "red": _create_material(
            stage, "/World/Materials/Red", (0.72, 0.04, 0.03), 0.35
        ),
        "green": _create_material(
            stage, "/World/Materials/Green", (0.04, 0.50, 0.10), 0.32
        ),
        "blue": _create_material(
            stage, "/World/Materials/Blue", (0.04, 0.16, 0.72), 0.30
        ),
        "yellow": _create_material(
            stage, "/World/Materials/Yellow", (0.90, 0.55, 0.03), 0.38
        ),
        "white": _create_material(
            stage, "/World/Materials/White", (0.82, 0.84, 0.88), 0.42
        ),
        "metal": _create_material(
            stage,
            "/World/Materials/Metal",
            (0.50, 0.52, 0.55),
            0.22,
            metallic=0.75,
        ),
        "dark": _create_material(
            stage, "/World/Materials/Dark", (0.06, 0.07, 0.08), 0.55
        ),
    }

    _create_cube(
        stage,
        "/World/Ground",
        (0.0, 0.0, -0.025),
        (config.ground_size, config.ground_size, 0.05),
        materials["ground"],
    )

    _create_cube(
        stage,
        "/World/Table/Top",
        (0.0, 0.0, config.table_top_center_z),
        (
            config.table_top_size_x,
            config.table_top_size_y,
            config.table_top_thickness,
        ),
        materials["wood"],
    )

    leg_height = (
        config.table_top_center_z - config.table_top_thickness * 0.5
    )
    leg_z = leg_height * 0.5
    leg_x = config.table_top_size_x * 0.5 - 0.10
    leg_y = config.table_top_size_y * 0.5 - 0.10
    for index, (x, y) in enumerate(
        ((-leg_x, -leg_y), (leg_x, -leg_y), (-leg_x, leg_y), (leg_x, leg_y))
    ):
        _create_cube(
            stage,
            f"/World/Table/Leg_{index}",
            (x, y, leg_z),
            (0.08, 0.08, leg_height),
            materials["wood"],
        )

    surface_z = (
        config.table_top_center_z + config.table_top_thickness * 0.5
    )

    # 麦片盒。
    _create_cube(
        stage,
        "/World/Objects/CerealBox",
        (-0.38, 0.17, surface_z + 0.18),
        (0.24, 0.11, 0.36),
        materials["yellow"],
        (0.0, 0.0, -12.0),
    )

    # 罐头。
    _create_cylinder(
        stage,
        "/World/Objects/FoodCan",
        (0.02, 0.25, surface_z + 0.095),
        0.07,
        0.19,
        materials["metal"],
    )

    # 瓶子。
    _define_xform(stage, "/World/Objects/Bottle")
    _create_cylinder(
        stage,
        "/World/Objects/Bottle/Body",
        (0.39, 0.18, surface_z + 0.13),
        0.065,
        0.26,
        materials["green"],
    )
    _create_sphere(
        stage,
        "/World/Objects/Bottle/Shoulder",
        (0.39, 0.18, surface_z + 0.265),
        0.066,
        materials["green"],
        (1.0, 1.0, 0.55),
    )
    _create_cylinder(
        stage,
        "/World/Objects/Bottle/Neck",
        (0.39, 0.18, surface_z + 0.315),
        0.027,
        0.10,
        materials["green"],
    )
    _create_cylinder(
        stage,
        "/World/Objects/Bottle/Cap",
        (0.39, 0.18, surface_z + 0.372),
        0.031,
        0.026,
        materials["dark"],
    )

    # 杯子与把手。
    _define_xform(stage, "/World/Objects/Mug")
    _create_cylinder(
        stage,
        "/World/Objects/Mug/Body",
        (-0.18, -0.20, surface_z + 0.065),
        0.082,
        0.13,
        materials["blue"],
    )
    _create_cube(
        stage,
        "/World/Objects/Mug/HandleTop",
        (-0.18, -0.302, surface_z + 0.105),
        (0.035, 0.09, 0.025),
        materials["blue"],
    )
    _create_cube(
        stage,
        "/World/Objects/Mug/HandleBottom",
        (-0.18, -0.302, surface_z + 0.035),
        (0.035, 0.09, 0.025),
        materials["blue"],
    )
    _create_cube(
        stage,
        "/World/Objects/Mug/HandleOuter",
        (-0.18, -0.347, surface_z + 0.07),
        (0.035, 0.025, 0.095),
        materials["blue"],
    )

    # 苹果。
    _create_sphere(
        stage,
        "/World/Objects/Apple",
        (0.18, -0.20, surface_z + 0.068),
        0.068,
        materials["red"],
        (1.0, 1.0, 0.92),
    )
    _create_cylinder(
        stage,
        "/World/Objects/AppleStem",
        (0.18, -0.20, surface_z + 0.137),
        0.008,
        0.035,
        materials["wood"],
        (8.0, 0.0, 0.0),
    )

    # 碗的几何代理。
    _define_xform(stage, "/World/Objects/Bowl")
    _create_cylinder(
        stage,
        "/World/Objects/Bowl/Outer",
        (0.48, -0.22, surface_z + 0.045),
        0.14,
        0.09,
        materials["white"],
    )
    _create_cylinder(
        stage,
        "/World/Objects/Bowl/Inner",
        (0.48, -0.22, surface_z + 0.094),
        0.105,
        0.012,
        materials["dark"],
    )

    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
    dome.CreateIntensityAttr(500.0)
    dome.CreateColorAttr(Gf.Vec3f(0.95, 0.97, 1.0))

    key = UsdLux.DistantLight.Define(stage, "/World/Lights/Key")
    key.CreateIntensityAttr(2600.0)
    key.CreateAngleAttr(1.0)
    _set_xform(
        key.GetPrim(),
        (0.0, 0.0, 3.5),
        rotation_xyz_deg=(-38.0, 20.0, 24.0),
    )

    fill = UsdLux.SphereLight.Define(stage, "/World/Lights/Fill")
    fill.CreateIntensityAttr(18000.0)
    fill.CreateRadiusAttr(0.35)
    fill.CreateColorAttr(Gf.Vec3f(1.0, 0.80, 0.68))
    _set_xform(fill.GetPrim(), (-1.4, -1.0, 2.3))

    return {
        "table": "/World/Table",
        "cereal_box": "/World/Objects/CerealBox",
        "food_can": "/World/Objects/FoodCan",
        "bottle": "/World/Objects/Bottle",
        "mug": "/World/Objects/Mug",
        "apple": "/World/Objects/Apple",
        "bowl": "/World/Objects/Bowl",
    }
