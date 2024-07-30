import matplotlib

LAYOUTS = {
    "icml": dict(
        # ICML
        TEXT_WIDTH=6.00117,
        COL_WIDTH=3.25063,
        TEXT_HEIGHT=8.50166,
        FOOTNOTE_SIZE=8,
        SCRIPT_SIZE=7,
    ),
    "poster": dict(
        # Poster
        TEXT_WIDTH=12.8838,
        COL_WIDTH=12.8838,
        TEXT_HEIGHT=27.04193,
        FOOTNOTE_SIZE=30,
        SCRIPT_SIZE=23,
    ),
}

# cmfont = font_manager.FontProperties(fname=mpl.get_data_path() + '/fonts/ttf/cmr10.ttf')
# FONT_NAME = cmfont.get_name()
# FONT_NAME = "Times New Roman"
FONT_NAME = "Avenir Next Condensed"


def get_mpl_rcParams(width_percent, height_percent, single_col=False, poster=False):
    layout = LAYOUTS["poster"] if poster else LAYOUTS["icml"]
    params = {
        "text.usetex": False,
        "font.size": layout["SCRIPT_SIZE"],
        "font.family": "serif",
        "font.serif": FONT_NAME,
        "mathtext.fontset": "cm",
        "lines.linewidth": 4 if poster else 1,
        "axes.linewidth": 1 if poster else 0.5,
        "axes.titlesize": layout["FOOTNOTE_SIZE"],
        "axes.labelsize": layout["SCRIPT_SIZE"],
        "axes.unicode_minus": False,
        "axes.formatter.use_mathtext": True,
        "legend.frameon": False,
        "legend.fontsize": layout["SCRIPT_SIZE"],
        "legend.handlelength": 1,
        # "legend.linewidth": 3 if poster else 2,
        "xtick.major.size": 4 if poster else 1.5,
        "ytick.major.size": 4 if poster else 1.5,
        "xtick.major.width": 2 if poster else 0.5,
        "ytick.major.width": 2 if poster else 0.5,
    }

    w = width_percent * (layout["COL_WIDTH"] if single_col else layout["TEXT_WIDTH"])
    h = height_percent * layout["TEXT_HEIGHT"]

    return params, w, h
