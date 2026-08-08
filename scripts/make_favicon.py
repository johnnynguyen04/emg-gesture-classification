"""Draw the tab icon: brand gradient tile with a white EMG trace."""
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "streamlit_app" / "assets" / "favicon.png"

S = 256
TOP = (0, 92, 149)      # deep blue
BOTTOM = (27, 168, 225) # sky blue


def main() -> None:
    grad = Image.new("RGBA", (S, S))
    d = ImageDraw.Draw(grad)
    for y in range(S):
        t = y / (S - 1)
        color = tuple(round(TOP[i] + (BOTTOM[i] - TOP[i]) * t) for i in range(3))
        d.line([(0, y), (S, y)], fill=(*color, 255))

    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, S - 1, S - 1], radius=58, fill=255)

    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    img.paste(grad, (0, 0), mask)

    trace = [(30, 128), (78, 128), (100, 70), (132, 186), (156, 92), (174, 128), (226, 128)]
    ImageDraw.Draw(img).line(trace, fill=(255, 255, 255, 255), width=17, joint="curve")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT)
    print(f"saved {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
