"""Compare population and business growth, drawn from https://ocw.mit.edu/courses/15-988-system-dynamics-self-study-fall-1998-spring-1999/7ac2f07c6b562211becc8afb0102cf88_modeling2.pdf

See urban_growth_model notebook"""

import reno as r

business_sector = r.Model("business_sector")
with business_sector:
    structures = r.Stock(init=1000, doc="Number of business structures in the city.")
    construction = r.Flow(
        doc="Rate of construction of business structures. It is affected by the number of already existing business structures, a normal construction fraction, and the availability of land and labor."
    )
    demolition = r.Flow(doc="Rate of demolition of business structures.")

    average_structure_lifetime = r.Variable(
        50, doc="Average lifetime of a business structure."
    )
    construction_fraction = r.Variable(
        0.02,
        doc="Normal rate of construction of business structure per existing business structure.",
    )
    jobs = r.Variable(
        doc="Number of jobs provided by the existing business structures. It is the product of the number of business structures and the average number of jobs per structure."
    )
    jobs_per_structure = r.Variable(
        20, doc="The number of jobs provided by each structure."
    )
    labor_availability = r.Variable(
        doc="The ratio between the labor force and the number of available jobs."
    )
    land_area = r.Variable(
        5000, doc="Total land area available for commercial development."
    )
    land_fraction_occupied = r.Variable(
        doc="The fraction of commercial land that has already been developed."
    )
    land_per_structure = r.Variable(
        1, doc="The amount of land required by each business structure."
    )

    labor_availability_multiplier = r.Variable(
        r.interpolate(
            labor_availability,
            [0.0, 0.2, 0.4, 0.6, 0.8, 1.00, 1.20, 1.4, 1.6, 1.8, 2.0],
            [0.05, 0.105, 0.225, 0.36, 0.54, 0.84, 1.24, 2.36, 3.34, 3.86, 4.0],
        )
    )

    land_availability_multiplier = r.Variable(
        r.interpolate(
            land_fraction_occupied,
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            [1.0, 2.3, 2.98, 3.34, 3.48, 3.5, 3.44, 3.12, 2.3, 1.0, 0.0],
        )
    )

    structures += construction
    structures -= demolition
    construction.eq = (
        structures
        * construction_fraction
        * labor_availability_multiplier
        * land_availability_multiplier
    )
    demolition.eq = structures / average_structure_lifetime

    jobs.eq = structures * jobs_per_structure
    land_fraction_occupied.eq = structures * land_per_structure / land_area


population_sector = r.Model("population_sector")
with population_sector:
    population = r.Stock(
        init=50_000, doc="The number of people living in the urban area."
    )
    in_migration = r.Flow(
        doc="The number of people who move into the urban area each year. It is affected by the current population, a normal fraction of in-migration, and the availability of jobs."
    )
    births = r.Flow(doc="The number of people born in the area per year.")
    out_migration = r.Flow(
        doc="The number of people who leave the urban area each year."
    )
    deaths = r.Flow(doc="Number of people who die each year.")

    average_lifetime = r.Variable(
        66.7,
        doc="The average lifetime of a person living in the urban area is approximately 67 years.",
    )
    birth_fraction = r.Variable(
        0.015, doc="The fraction of the population that reproduces each year."
    )
    in_migration_normal = r.Variable(
        0.08,
        doc="The fraction of the population that immigrates each year under normal conditions.",
    )
    labor_force = r.Variable(
        doc="The number of people who are eligible to work. It is a constant fraction of the population."
    )
    labor_participation_fraction = r.Variable(
        0.35,
        doc="The fraction of the total population that is willing and able to work.",
    )
    out_migration_fraction = r.Variable(
        0.08, doc="The fraction of the population that emigrates each year."
    )

    job_attractiveness_multiplier = r.Variable(
        r.interpolate(
            business_sector.labor_availability,
            [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
            [4.0, 3.95, 3.82, 3.56, 2.86, 1.24, 0.64, 0.32, 0.18, 0.105, 0.075],
        ),
        doc="The multiplier shows the effect of labor availability on immigration. When there are many available jobs (labor availability is less than 1), people are inclined to move to the city. When there are not enough available jobs (labor availability is greater than 1), people tend not to immigrate to the urban area.",
    )

    population += in_migration
    population += births
    population -= out_migration
    population -= deaths

    in_migration.eq = population * in_migration_normal * job_attractiveness_multiplier
    births.eq = population * birth_fraction
    out_migration.eq = population * out_migration_fraction
    deaths.eq = population / average_lifetime

    labor_force.eq = population * labor_participation_fraction

business_sector.labor_availability.eq = (
    population_sector.labor_force / business_sector.jobs
)

urban_growth = r.Model("urban_growth")
urban_growth.business = business_sector
urban_growth.population = population_sector
