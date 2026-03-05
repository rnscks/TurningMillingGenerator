from core.turning.features import create_stock, StockInfo
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax2



stock = create_stock(StockInfo(height=10, radius=5))

from OCC.Display.SimpleGui import  init_display


a, b, c, d = init_display()
a.DisplayShape(stock)
a.FitAll()
b()