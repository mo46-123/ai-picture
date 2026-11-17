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
    LipShade("peach", "珊瑚蜜桃", (240, 112, 124)),
    LipShade("berry", "柔莓粉", (220, 106, 136)),
    LipShade("milktea", "奶茶豆沙", (186, 108, 98)),
    LipShade("strawberry", "草莓咬唇", (216, 48, 72)),
    LipShade("rose", "冷调玫瑰", (158, 70, 96)),
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
    blush_out: float       # 0 = under the pupil, larger = toward the cheekbone
    aegyo: float
    eye_brighten: float
    teeth: float
    brow_rgb: tuple[int, int, int]
    brow: float
    shadow_rgb: tuple[int, int, int]
    shadow: float          # eyeshadow
    liner: float           # eyeliner
    liner_tail: float      # tail length (× eye distance)
    liner_drop: float      # + = tail points down (puppy eye), − = lifted
    shadow_outer: float    # smoky outer corner
    blush_nose: float      # blush on the nose tip
    brow_shape: float      # + straight brow (一字眉), − lifted tail
    lash: float            # drawn upper lashes
    lip_plump: float       # 丰唇
    sculpt: float          # 修容高光
    lens: float            # 美瞳
    hair: float            # deeper, glossier hair around the face
    relief: float          # 立体: the face's own light and shade, restored and deepened
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
        smooth=0.7, texture=0.85, whiten=0.42, tone_a=1.2, tone_b=-4.5, glow=0.8, matte=0.0,
        slim=0.7, eyes=0.5, nose=0.5,
        lip="berry", lip_intensity=1.0, lip_gradient=0.35, lip_gloss=1.0,
        blush_rgb=(246, 150, 174), blush=0.8, blush_height=0.55, blush_spread=1.45, blush_out=0.2,
        aegyo=0.9, eye_brighten=0.8, teeth=0.6,
        brow_rgb=(122, 94, 80), brow=0.3, shadow_rgb=(240, 156, 150), shadow=0.7, liner=0.7,
        liner_tail=0.07, liner_drop=0.0, shadow_outer=0.0, blush_nose=0.0, brow_shape=0.3, lash=0.6, lip_plump=0.6,
        sculpt=1.0, lens=1.0, hair=1.0, relief=1.0,
        dark_circles=0.75, folds=0.6,
        grade=Grade(exposure=0.04, contrast=0.22, highlights=-0.08, shadows=-0.28,
                    temperature=-0.22, tint=0.03, saturation=-0.10, vibrance=0.08, fade=0.0,
                    shadow_tint=(215, 0.25), highlight_tint=(350, 0.20),
                    green_shift=20, green_sat=-0.20, glow=0.08),
    ),
    Preset(
        id="cream_skin", name="奶油肌", name_en="Cream Skin",
        tagline="柔雾哑光，温柔奶茶调",
        description="细腻哑光的奶油肌，低对比的暖杏色和奶茶色滤镜，配奶茶豆沙雾面唇。",
        swatch=("#FFF1DD", "#F1CBB3"),
        smooth=0.75, texture=0.8, whiten=0.38, tone_a=0.6, tone_b=-1.0, glow=0.2, matte=0.45,
        slim=0.55, eyes=0.4, nose=0.4,
        lip="milktea", lip_intensity=1.0, lip_gradient=0.05, lip_gloss=0.0,
        blush_rgb=(226, 150, 120), blush=0.75, blush_height=0.75, blush_spread=1.25, blush_out=0.12,
        aegyo=0.6, eye_brighten=0.6, teeth=0.5,
        brow_rgb=(128, 100, 84), brow=0.35, shadow_rgb=(190, 132, 104), shadow=0.85, liner=0.3,
        liner_tail=0.03, liner_drop=0.0, shadow_outer=0.25, blush_nose=0.0, brow_shape=0.3, lash=0.8, lip_plump=0.4,
        sculpt=0.6, lens=0.6, hair=0.3, relief=0.7,
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
        smooth=0.62, texture=0.85, whiten=0.38, tone_a=1.6, tone_b=-2.0, glow=0.4, matte=0.0,
        slim=0.6, eyes=0.55, nose=0.45,
        lip="strawberry", lip_intensity=1.0, lip_gradient=0.95, lip_gloss=0.5,
        blush_rgb=(240, 130, 150), blush=0.85, blush_height=0.4, blush_spread=1.1, blush_out=0.14,
        aegyo=1.0, eye_brighten=0.75, teeth=0.6,
        brow_rgb=(128, 98, 84), brow=0.3, shadow_rgb=(236, 140, 150), shadow=0.6, liner=0.6,
        liner_tail=0.06, liner_drop=0.035, shadow_outer=0.0, blush_nose=0.8, brow_shape=0.6, lash=1.0, lip_plump=0.6,
        sculpt=0.8, lens=0.9, hair=0.5, relief=0.85,
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
        smooth=0.65, texture=0.85, whiten=0.55, tone_a=-0.3, tone_b=-4.0, glow=0.2, matte=0.3,
        slim=0.65, eyes=0.4, nose=0.5,
        lip="rose", lip_intensity=1.0, lip_gradient=0.1, lip_gloss=0.1,
        blush_rgb=(214, 164, 160), blush=0.25, blush_height=0.7, blush_spread=0.9, blush_out=0.12,
        aegyo=0.0, eye_brighten=0.5, teeth=0.6,
        brow_rgb=(84, 72, 70), brow=0.4, shadow_rgb=(150, 118, 118), shadow=0.8, liner=0.85,
        liner_tail=0.09, liner_drop=-0.03, shadow_outer=1.0, blush_nose=0.0, brow_shape=-0.5, lash=0.9, lip_plump=0.3,
        sculpt=0.9, lens=0.8, hair=0.8, relief=1.0,
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
