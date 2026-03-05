from core.turning.features import create_stock, create_step_cut, create_groove_cut, apply_groove_cut

from OCC.Display.SimpleGui import init_display

stock = create_stock(height=10, radius=5)

step_cut = apply_groove_cut(stock, 0, 5, 10, 4)

a, b, c, d = init_display()
a.DisplayShape(step_cut)
a.FitAll()

b()