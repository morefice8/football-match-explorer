"""Shared coordinate rules for Match Analysis pitch visualizations.

Contract:
- processed Match Analysis coordinates are team-relative;
- x=0 is the team's own goal and x=100 is the opponent goal;
- every analytical pitch is therefore displayed left-to-right for both teams;
- home/away may change labels or palette, never geometry.
"""

ATTACKING_DIRECTION_LABEL = "Attacking →"


def orient_point(x, y, *, is_away=False):
    """Return a point unchanged under the Match Analysis coordinate contract."""
    del is_away
    return x, y


def orient_segment(x, y, end_x, end_y, *, is_away=False):
    """Return a segment unchanged under the Match Analysis coordinate contract."""
    del is_away
    return x, y, end_x, end_y


def add_matplotlib_attacking_direction(
    ax,
    color='#27445b',
    *,
    x=0.985,
    y=0.985,
):
    """Add the canonical attacking-direction label to a Matplotlib pitch."""
    ax.text(
        x,
        y,
        ATTACKING_DIRECTION_LABEL,
        transform=ax.transAxes,
        ha='right',
        va='top',
        fontsize=10,
        fontweight='bold',
        color=color,
        zorder=20,
    )
    return ax
