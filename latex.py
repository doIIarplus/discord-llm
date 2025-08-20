import matplotlib.pyplot as plt
import matplotlib
import re
import os
import uuid

matplotlib.use('Agg')

def is_simple_latex(expr):
    """
    Heuristically determine if LaTeX expression is simple (e.g., single variable, subscript, superscript)
    """
    # Remove LaTeX commands like \alpha or \frac
    cleaned = re.sub(r'\\[a-zA-Z]+', '', expr)

    # Remove braces, carets, underscores
    cleaned = re.sub(r'[{}\^\_]', '', cleaned)

    # Count how many visible math characters remain
    visible_tokens = re.findall(r'[a-zA-Z0-9]', cleaned)
    return len(visible_tokens) <= 3

def render_latex_to_image(latex, output_dir='latex_images'):
    print(f"Rendering latex {latex}")
    latex = latex.strip()
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{uuid.uuid4().hex}.png"
    filepath = os.path.join(output_dir, filename)

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "mathtext.fontset": "cm",  # Computer Modern
        "mathtext.rm": "serif",    # Roman serif font
        "text.latex.preamble": "\\usepackage{amsmath,amssymb}",
    })

    fig = plt.figure()
    # Transparent background
    fig.patch.set_alpha(0.0)

    # White text
    font_size = 5 if is_simple_latex(latex) else 9
    text = fig.text(0, 0, f'${latex}$', fontsize=font_size, color='white')

    # Resize to bounding box
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = text.get_window_extent(renderer).transformed(fig.dpi_scale_trans.inverted())

    width, height = bbox.width * 1.05, bbox.height * 1.05
    fig.set_size_inches(width, height)

    text.set_position((0.025, 0.025))

    # Turn off axes
    plt.axis('off')

    # Save with transparent background
    plt.savefig(
        filepath,
        dpi=300,
        bbox_inches='tight',
        pad_inches=0.01,
        transparent=True
    )
    plt.close(fig)
    return filepath

def split_text_preserve_limit(text, limit=1900):
    # Splits long strings into chunks of up to `limit` characters
    return [text[i:i+limit] for i in range(0, len(text), limit)]

def split_text_and_latex(input_string, max_length=1900):
    # Matches both inline ($...$) and display ($$...$$) math
    pattern = re.compile(
        r'''
            (?<!\\)                            # Not preceded by a backslash
            (                                  # Group 1: the full match
                \$\$(.*?)\$\$                  #   Group 2: $$...$$
                | \$([^$].*?)\$                #   Group 3: $...$ (avoid $$$ edge case)
                | \\\( (.*?) \\\)              #   Group 4: \(...\)
                | \\\[ (.*?) \\]               #   Group 5: \[...\]
            )
        ''',
        re.DOTALL | re.VERBOSE
    )
        
    result = []
    last_index = 0

    for match in pattern.finditer(input_string):
        start, end = match.span()
        prefix = input_string[last_index:start]

        # Add preceding plain text, split if needed
        if prefix:
            result.extend(split_text_preserve_limit(prefix, max_length))

        # Determine which group matched for the LaTeX content
        latex_expr = next(g for g in match.groups()[1:] if g is not None)

        try:
            latex_img = render_latex_to_image(latex_expr)
            result.append({'latex': latex_expr, 'image': latex_img})
        except Exception as e:
            print("Error rendering LaTeX:", e)
            result.extend(split_text_preserve_limit(latex_expr, max_length))

        last_index = end

    # Handle trailing text
    suffix = input_string[last_index:]
    if suffix:
        result.extend(split_text_preserve_limit(suffix, max_length))

    return result