from texttable import Texttable
import numpy as np
import latextable


gpl_spi_cooking = [12, 32.5, 12, 24.3]
mappo_cooking = [12, 32.5, 12, 24.3]
moop_ca_cooking = [12, 32.5, 12, 24.3]
moop_fr_cooking = [12, 32.5, 12, 24.3]

gpl_spi_lbs = [12, 32.5, 12, 24.3]
mappo_lbs = [12, 32.5, 12, 24.3]
moop_ca_lbs = [12, 32.5, 12, 24.3]
moop_fr_lbs = [12, 32.5, 12, 24.3]

gpl_spi_elbs = [12, 32.5, 12, 24.3]
mappo_elbs = [12, 32.5, 12, 24.3]
moop_ca_elbs = [12, 32.5, 12, 24.3]
moop_fr_elbs = [12, 32.5, 12, 24.3]

gpl_spi_wolfpack = [12, 32.5, 12, 24.3]
mappo_wolfpack = [12, 32.5, 12, 24.3]
moop_ca_wolfpack = [12, 32.5, 12, 24.3]
moop_fr_wolfpack = [12, 32.5, 12, 24.3]

environments = ["CookingZoo", "Lbs", "Extended LBS", "Wolfpack"]

header = ["Env.", "MoOP-FR", "MoOP-CA", "GPL-SPI", "MAPPO"]

pm = u"\u00B1"

rows = [header,
        [environments[0], f"{np.mean(moop_fr_cooking)}{pm}{np.std(moop_fr_cooking):.2f}",
         f"{np.mean(moop_ca_cooking)}{pm}{np.std(moop_ca_cooking):.2f}",
         f"{np.mean(gpl_spi_cooking)}{pm}{np.std(gpl_spi_cooking):.2f}",
         f"{np.mean(mappo_cooking)}{pm}{np.std(mappo_cooking):.2f}"],
        [environments[1], f"{np.mean(moop_fr_lbs)}{pm}{np.std(moop_fr_lbs):.2f}",
         f"{np.mean(moop_ca_lbs)}{pm}{np.std(moop_ca_lbs):.2f}",
         f"{np.mean(gpl_spi_lbs)}{pm}{np.std(gpl_spi_lbs):.2f}",
         f"{np.mean(mappo_lbs)}{pm}{np.std(mappo_lbs):.2f}"],
        [environments[2], f"{np.mean(moop_fr_elbs)}{pm}{np.std(moop_fr_elbs):.2f}",
         f"{np.mean(moop_ca_elbs)}{pm}{np.std(moop_ca_elbs):.2f}",
         f"{np.mean(gpl_spi_elbs)}{pm}{np.std(gpl_spi_elbs):.2f}",
         f"{np.mean(mappo_elbs)}{pm}{np.std(mappo_elbs):.2f}"],
        [environments[3], f"{np.mean(moop_fr_wolfpack)}{pm}{np.std(moop_fr_wolfpack):.2f}",
         f"{np.mean(moop_ca_wolfpack)}{pm}{np.std(moop_ca_wolfpack):.2f}",
         f"{np.mean(gpl_spi_wolfpack)}{pm}{np.std(gpl_spi_wolfpack):.2f}",
         f"{np.mean(mappo_wolfpack)}{pm}{np.std(mappo_wolfpack):.2f}"]]

table = Texttable()
table.set_cols_align(["c"] * 5)
table.set_deco(Texttable.HEADER | Texttable.VLINES)
table.add_rows(rows)

print('\nTexttable Table:')
print(table.draw())

print(latextable.draw_latex(table, caption="A comparison of rocket features."))