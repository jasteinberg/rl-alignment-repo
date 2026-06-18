"""Generate self-contained true/false datasets (statement,label) locally.
Stand-ins for the Geometry-of-Truth larger_than / cities sets while offline.
Scaffolded with assistance from Claude (Anthropic)."""
import os, csv, random
random.seed(0)
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "geometry_of_truth")
os.makedirs(DATA, exist_ok=True)

rows, seen = [], set()
while len(rows) < 1200:
    a, b = random.randint(1, 99), random.randint(1, 99)
    if a == b or (a, b) in seen:
        continue
    seen.add((a, b)); rows.append((f"{a} is larger than {b}.", int(a > b)))
with open(f"{DATA}/larger_than.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["statement", "label"]); w.writerows(rows)

pairs = [("Paris","France"),("Tokyo","Japan"),("Cairo","Egypt"),("Madrid","Spain"),
("Rome","Italy"),("Berlin","Germany"),("Lisbon","Portugal"),("Ottawa","Canada"),
("Canberra","Australia"),("Nairobi","Kenya"),("Bangkok","Thailand"),("Hanoi","Vietnam"),
("Athens","Greece"),("Oslo","Norway"),("Helsinki","Finland"),("Vienna","Austria"),
("Warsaw","Poland"),("Budapest","Hungary"),("Dublin","Ireland"),("Lima","Peru"),
("Santiago","Chile"),("Bogota","Colombia"),("Seoul","South Korea"),("Jakarta","Indonesia"),
("Manila","Philippines"),("Ankara","Turkey"),("Tehran","Iran"),("Baghdad","Iraq"),
("Amman","Jordan"),("Havana","Cuba"),("Quito","Ecuador"),("Caracas","Venezuela"),
("Stockholm","Sweden"),("Copenhagen","Denmark"),("Brussels","Belgium"),("Bern","Switzerland"),
("Prague","Czechia"),("Moscow","Russia"),("Kabul","Afghanistan"),("Wellington","New Zealand")]
countries = sorted({c for _, c in pairs})
crows = []
for city, country in pairs:
    crows.append((f"The city of {city} is in {country}.", 1))
    wrong = random.choice([c for c in countries if c != country])
    crows.append((f"The city of {city} is in {wrong}.", 0))
random.shuffle(crows)
with open(f"{DATA}/cities.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["statement", "label"]); w.writerows(crows)
print("larger_than rows:", len(rows), "| cities rows:", len(crows))
