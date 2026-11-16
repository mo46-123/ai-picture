"""Prompts for the generative "AI 韩系写真" mode.

Rule #1 for a retouching app: the result must still look like the user.
Every prompt therefore states the style *and* what must not change.
Chinese prompts go to Chinese models (百炼 / 火山), English ones to Gemini.
"""

from __future__ import annotations

KEEP_ZH = (
    "严格保持照片中人物的身份：五官形状、脸型、发型、表情、姿势和服装都不要改变，"
    "必须一眼看出是同一个人，只是经过韩系精修后更好看。保留真实的皮肤纹理，不要塑料感，"
    "不要过度磨皮，不要改变年龄和体型，不要添加文字、水印或边框。"
)
KEEP_EN = (
    "Strictly preserve the person's identity: keep the facial features, face shape, hairstyle, "
    "expression, pose and clothing unchanged — it must clearly be the same person, just "
    "beautifully retouched in a Korean style. Keep realistic skin texture (no plastic look, no "
    "over-smoothing), do not change age or body shape, and add no text, watermark or border."
)

STYLE_ZH = {
    "glass_skin": "把这张人像修成韩系「水光肌」风格：皮肤清透、自带水润光泽，像刚敷完面膜一样透亮；"
                  "淡淡的蜜桃色腮红，蜜桃珊瑚色水光唇；画面明亮通透、低饱和、略带粉调，柔和自然光，像韩国美妆广告。"
                  "保持原图构图和背景。",
    "cream_skin": "把这张人像修成韩系「奶油肌」风格：细腻柔雾的哑光肌肤，温柔的暖杏色和奶茶色调，低对比、低饱和，"
                  "奶茶豆沙色雾面唇，淡淡的杏色腮红，温暖柔和的午后室内光。保持原图构图和背景。",
    "kdrama": "把这张人像修成「韩剧女主」氛围感：柔焦和淡淡的光晕，韩剧画面般温柔的逆光与暖粉色调，轻微胶片质感；"
              "草莓色渐变咬唇妆，眼下位置的粉色腮红，自然的卧蚕，眼神清澈有光。保持原图构图和背景。",
    "cool_tone": "把这张人像修成韩系「清冷感」：冷白透亮的肤色，低饱和、偏冷的蓝灰色调，干净高级的杂志感；"
                 "冷调玫瑰色唇妆，极淡的腮红，清透干净的底妆。保持原图构图和背景。",
    "id_photo": "把这张人像修成「韩式证件照」（韩国照相馆精修风格）：正面、肩部以上构图，背景换成干净柔和的浅灰蓝纯色，"
                "柔和均匀的棚拍灯光，皮肤干净无瑕但保留质感，自然的韩系淡妆，碎发整理整齐，整体精致高级。",
}
STYLE_EN = {
    "glass_skin": "Retouch this portrait in the Korean 'glass skin' style: clear, dewy, luminous skin with a "
                  "natural sheen, a soft peach blush and glossy peach-coral lips; a bright, airy, slightly pink, "
                  "low-saturation look with soft natural light, like a Korean beauty ad. Keep the original "
                  "framing and background.",
    "cream_skin": "Retouch this portrait in the Korean 'cream skin' style: soft-focus velvety matte skin, gentle "
                  "warm apricot and milk-tea tones, low contrast and saturation, muted rosy-nude matte lips, a "
                  "light apricot blush and warm afternoon indoor light. Keep the original framing and background.",
    "kdrama": "Retouch this portrait with a K-drama heroine mood: soft focus and a gentle glow, warm pink "
              "backlight like a Korean drama still, subtle film grain; strawberry gradient lips, pink under-eye "
              "blush, natural aegyo-sal and bright clear eyes. Keep the original framing and background.",
    "cool_tone": "Retouch this portrait in a cool, elegant Korean style: porcelain cool-toned skin, a desaturated "
                 "blue-gray palette, clean magazine look, cool rose lips, a barely-there blush and a clear, "
                 "fresh base. Keep the original framing and background.",
    "id_photo": "Turn this portrait into a Korean studio ID photo: front-facing, head-and-shoulders framing, a clean "
                "soft light gray-blue solid background, soft even studio lighting, flawless but realistic skin, "
                "natural Korean makeup and neatly tidied flyaway hair; refined and polished.",
}

NEGATIVE_ZH = "换脸, 不像本人, 改变五官, 塑料感皮肤, 过度磨皮, 蜡像感, 卡通, 动漫, 畸形, 多余的手指, 文字, 水印, 低分辨率, 过饱和"
NEGATIVE_EN = ("different person, changed facial features, plastic skin, over-smoothed, waxy, cartoon, anime, "
               "deformed, extra fingers, text, watermark, low resolution, oversaturated")


def build_prompt(style_id: str, language: str = "zh") -> tuple[str, str]:
    """Return (prompt, negative_prompt) for a preset id."""
    if language == "en":
        style = STYLE_EN.get(style_id, STYLE_EN["glass_skin"])
        return f"{style} {KEEP_EN}", NEGATIVE_EN
    style = STYLE_ZH.get(style_id, STYLE_ZH["glass_skin"])
    return f"{style}{KEEP_ZH}", NEGATIVE_ZH
