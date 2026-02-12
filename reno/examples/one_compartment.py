"""One-Compartment model of repeated doses, drawn from https://assets.press.princeton.edu/chapters/s2_10291.pdf

See one_compartment notebook"""

import reno as r

t = r.TimeRef()
one_compartment_model = r.Model("one_compartment_model", steps=168)
with one_compartment_model:
    drug_in_system = r.Stock(doc='mass of medication in the ("person\'s blood serum"?)')
    ingested = r.Flow(doc="Pulsed inflow of medication when dosage is taken.")
    eliminated = r.Flow(doc="Rate of change of drug leaving the system.")

    absorption_fraction = r.Variable(0.12)
    dosage = r.Variable(100 * 1000, doc="Dosage is 100 * 1000 micrograms")
    start = r.Variable(0, doc="Timestep of first dosage. (in hours)")
    interval = r.Variable(8, doc="Timesteps between each dosage. (in hours)")

    volume = r.Variable(3000, doc="Volume of blood serum, 3000 mL")
    concentration = r.Variable(drug_in_system / volume)
    half_life = r.Variable(22, doc="Half-life of medication. (in hours)")
    elimination_constant = r.Variable(-r.log(0.5) / half_life)

    eliminated.eq = elimination_constant * drug_in_system
    ingested.eq = absorption_fraction * dosage * r.repeated_pulse(start, interval)

    ingested >> drug_in_system >> eliminated
