"""Korean-style presets (韩风预设) and lipstick shades (口红色号).

Each preset is a complete "recipe" for the local pipeline. Makeup values are the
*maximum* strength (the 妆容 slider at 100%); the UI starts at ``default_makeup``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .color import Grade


@dataclass(frozen=True)
class LipShade:
    id: str
    name: str
    rgb: tuple[int, int, int]

    @property
    def hex(self) -> str:
        return "#{:02X}{:02X}{:02X}".format(*self.rgb)


LIP_SHADES: dict[str, LipShade] = {s.id: s for s in [
    LipShade("peach", "蜜桃水光", (236, 112, 108)),
    LipShade("milktea", "奶茶豆沙", (196, 116, 108)),
    LipShade("strawberry", "草莓咬唇", (214, 58, 78)),
    LipShade("rose", "冷调玫瑰", (176, 92, 112)),
    LipShade("nude", "自然裸粉", (212, 124, 122)),
]}


@dataclass(frozen=True)
class Preset:
    id: str
    name: str
    name_en: str
    tagline: str
    description: str
    swatch: tuple[str, str]
    # skin
    smooth: float
    texture: float
    whiten: float
    tone_a: float          # Lab a* shift on skin (+ = rosier)
    tone_b: float          # Lab b* shift on skin (− = less yellow)
    glow: float            # glass-skin sheen
    matte: float           # velvet / matte finish
    # shape
    slim: float            # 瘦脸 + 小V下巴
    eyes: float            # 大眼
    nose: float            # 瘦鼻
    # makeup (at 100%)
    lip: str
    lip_intensity: float
    lip_gradient: float
    lip_gloss: float
    blush_rgb: tuple[int, int, int]
    blush: float
    blush_height: float
    blush_spread: float
    aegyo: float
    eye_brighten: float
    teeth: float
    brow_rgb: tuple[int, int, int]
    brow: float
    shadow_rgb: tuple[int, int, int]
    shadow: float          # eyeshadow
    liner: float           # eyeliner
    # skin clean-up (follows the 磨皮 slider)
    dark_circles: float
    folds: float
    # color
    grade: Grade
    default_makeup: float = 0.75
    default_filter: float = 0.7

    def ui_defaults(self) -> dict:
        return {
            "smooth": self.smooth, "whiten": self.whiten, "slim": self.slim, "eyes": self.eyes,
            "nose": self.nose, "makeup": self.default_makeup, "filter": self.default_filter,
            "lip": self.lip,
        }

    def to_public(self) -> dict:
        return {
            "id": self.id, "name": self.name, "name_en": self.name_en,
            "tagline": self.tagline, "description": self.description, "swatch": list(self.swatch),
            "defaults": self.ui_defaults(),
        }


PRESETS: dict[str, Preset] = {p.id: p for p in [
    Preset(
        id="glass_skin", name="水光肌", name_en="Glass Skin",
        tagline="清透发光，蜜桃水光唇",
        description="像刚敷完面膜一样水润透亮，高光点在额头、鼻梁和苹果肌，低饱和的通透粉调。",
        swatch=("#FFE3E6", "#CDEBF6"),
        smooth=0.65, texture=0.85, whiten=0.4, tone_a=1.0, tone_b=-2.5, glow=0.8, matte=0.0,
        slim=0.5, eyes=0.38, nose=0.4,
        lip="peach", lip_intensity=0.85, lip_gradient=0.55, lip_gloss=0.9,
        blush_rgb=(240, 156, 156), blush=0.7, blush_height=0.55, blush_spread=1.0,
        aegyo=0.55, eye_brighten=0.6, teeth=0.6,
        brow_rgb=(122, 94, 80), brow=0.5, shadow_rgb=(234, 158, 138), shadow=0.55, liner=0.35,
        dark_circles=0.75, folds=0.6,
        grade=Grade(exposure=0.10, contrast=0.02, highlights=-0.08, shadows=0.22,
                    temperature=-0.08, tint=0.06, saturation=-0.08, vibrance=0.08, fade=0.0,
                    shadow_tint=(215, 0.25), highlight_tint=(350, 0.20),
                    green_shift=20, green_sat=-0.20, glow=0.08),
    ),
    Preset(
        id="cream_skin", name="奶油肌", name_en="Cream Skin",
        tagline="柔雾哑光，温柔奶茶调",
        description="细腻哑光的奶油肌，低对比的暖杏色和奶茶色滤镜，配奶茶豆沙雾面唇。",
        swatch=("#FFF1DD", "#F1CBB3"),
        smooth=0.7, texture=0.85, whiten=0.35, tone_a=0.8, tone_b=-1.5, glow=0.15, matte=0.6,
        slim=0.45, eyes=0.32, nose=0.35,
        lip="milktea", lip_intensity=0.8, lip_gradient=0.3, lip_gloss=0.15,
        blush_rgb=(236, 160, 140), blush=0.7, blush_height=0.55, blush_spread=1.15,
        aegyo=0.45, eye_brighten=0.45, teeth=0.5,
        brow_rgb=(126, 98, 84), brow=0.5, shadow_rgb=(206, 148, 122), shadow=0.6, liner=0.3,
        dark_circles=0.75, folds=0.6,
        grade=Grade(exposure=0.18, contrast=-0.18, highlights=-0.30, shadows=0.25,
                    temperature=0.07, tint=0.04, saturation=-0.10, fade=0.05,
                    shadow_tint=(30, 0.20), highlight_tint=(40, 0.20),
                    green_shift=10, green_sat=-0.35, glow=0.12, grain=0.15),
    ),
    Preset(
        id="kdrama", name="韩剧女主", name_en="K-Drama",
        tagline="柔焦光晕，草莓咬唇妆",
        description="韩剧画面般的柔光和暖粉色调，眼下腮红、明显卧蚕和渐变咬唇，带一点胶片颗粒。",
        swatch=("#FFD5DE", "#FFE8C8"),
        smooth=0.62, texture=0.85, whiten=0.35, tone_a=1.2, tone_b=-2.0, glow=0.5, matte=0.0,
        slim=0.55, eyes=0.42, nose=0.45,
        lip="strawberry", lip_intensity=0.9, lip_gradient=0.85, lip_gloss=0.5,
        blush_rgb=(238, 150, 160), blush=0.75, blush_height=0.42, blush_spread=1.05,
        aegyo=0.75, eye_brighten=0.65, teeth=0.6,
        brow_rgb=(116, 88, 76), brow=0.55, shadow_rgb=(232, 146, 150), shadow=0.6, liner=0.4,
        dark_circles=0.75, folds=0.6,
        grade=Grade(exposure=0.12, contrast=-0.12, highlights=-0.20, shadows=0.20,
                    temperature=0.04, tint=0.08, saturation=-0.08, vibrance=0.05, fade=0.06,
                    shadow_tint=(200, 0.30), highlight_tint=(15, 0.22),
                    green_shift=15, green_sat=-0.25, glow=0.45, grain=0.22),
    ),
    Preset(
        id="cool_tone", name="清冷感", name_en="Cool Tone",
        tagline="冷白皮，低饱和高级感",
        description="冷白透亮的肤色和蓝灰色调，干净克制的杂志感，配冷调玫瑰唇。",
        swatch=("#E8EFF7", "#C8D3E4"),
        smooth=0.65, texture=0.85, whiten=0.42, tone_a=0.5, tone_b=-3.0, glow=0.3, matte=0.2,
        slim=0.55, eyes=0.35, nose=0.45,
        lip="rose", lip_intensity=0.8, lip_gradient=0.4, lip_gloss=0.3,
        blush_rgb=(226, 150, 166), blush=0.45, blush_height=0.5, blush_spread=0.95,
        aegyo=0.4, eye_brighten=0.55, teeth=0.6,
        brow_rgb=(104, 88, 82), brow=0.55, shadow_rgb=(188, 148, 158), shadow=0.5, liner=0.45,
        dark_circles=0.8, folds=0.6,
        grade=Grade(exposure=0.10, contrast=0.02, highlights=-0.15, shadows=0.18,
                    temperature=-0.15, tint=0.02, saturation=-0.12, fade=0.07,
                    shadow_tint=(215, 0.30), highlight_tint=(195, 0.15),
                    green_shift=25, green_sat=-0.45, glow=0.08, grain=0.12),
    ),
]}

DEFAULT_PRESET = "glass_skin"


def get_preset(preset_id: str | None) -> Preset:
    return PRESETS.get(preset_id or DEFAULT_PRESET, PRESETS[DEFAULT_PRESET])


def catalog() -> dict:
    return {
        "presets": [p.to_public() for p in PRESETS.values()],
        "lip_shades": [{"id": s.id, "name": s.name, "hex": s.hex} for s in LIP_SHADES.values()],
        "default": DEFAULT_PRESET,
    }


def grade_dict(preset: Preset) -> dict:
    return asdict(preset.grade)
