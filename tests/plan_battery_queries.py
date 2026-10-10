"""The query battery for the golden-plan regression test (see scripts/plan_battery.py).

Groups
------
unrelated     house / sold / commute / market / safety questions.  Their plans must NEVER change because of census,
              entity-matching or catalog work.  The committed golden was generated from the code BEFORE the census
              hardening, so passing means "zero drift", not merely "self-consistent".
near_miss     questions that merely CONTAIN census-sounding words ("population", "square miles", "area", "census tract")
              but belong to other concepts.  Same guarantee as ``unrelated``.
msa_context   existing metro / NRI / census questions.  They may improve (more places recognised) but only on purpose:
              a change shows up as a reviewable diff.
census        the questions this hardening was written for.

Adding a dataset?  Add two or three representative questions to a group, run ``python scripts/plan_battery.py --update``
and review the diff of ``tests/golden/plan_battery.json``: every changed plan outside your own dataset is a collision.
"""

GROUPS = {
    "unrelated": [
        "How many houses are in Pittsburgh?", "What is the average list price of houses in Denver?",
        "Which houses in my list have the highest walk scores?", "Show me the cheapest houses in Denver on a map",
        "How many active listings are there?", "What is the average sold price in Pittsburgh?", "Which cities have houses?",
        "Average bike score for homes in Austin", "How many houses are missing a walk score?", "median list price by city",
        "shortest bike commute", "average commute time to work", "zhvi home value trend", "market heat index in Austin",
        "price history for listings in Miami", "arm's length sale prices in Pittsburgh", "Find the shortest bike route from A to B",
        "number of crime incidents in Pittsburgh", "show my favorites", "are there any pending houses in Denver",
        "average transit score in Indianapolis", "list price of saved houses in Miami", "what is the weather like",
        "how many bedrooms does the cheapest house have", "crime density near downtown Pittsburgh",
    ],
    "near_miss": [
        "houses in Pittsburgh with population over 3000", "show houses within 5 square miles of downtown",
        "what is the lot size in square feet of the cheapest house", "population of the tract containing this house",
        "homes in the Pittsburgh area under 300000", "average price per square foot in Denver",
        "how many people commute by bike in Austin", "crime density in Pittsburgh", "average square feet of houses in Denver",
        "house price growth rate in Pittsburgh", "walk score near the Denver metro", "list homes by census tract in Miami",
        "which houses are in a high population neighborhood", "what is the area of the house with the most bedrooms",
        "compare list price of houses in Pittsburgh and Indianapolis", "the largest houses in Denver",
        "average density of listings per city", "population of the Atlantis metro", "what is the population of the state",
    ],
    "msa_context": [
        "Chart the average NRI risk score by MSA for Pittsburgh, Denver, Miami, and Austin",
        "Which metro has the highest riverine flood risk among Pittsburgh, Denver, Miami and Austin?",
        "What is the riverine flood risk for the Pittsburgh metro?", "Rank MSAs by population",
        "What is the combined MSA population of Pittsburgh and Denver?", "Is Pittsburgh part of a recognized CBSA?",
        "tract population in the Pittsburgh metro", "population density of Pittsburgh and Denver",
        "average hurricane risk for the Miami metro area", "largest metro by population", "smallest MSA",
    ],
    "census": [
        "Verify the land areas of Pittsburgh and Indianapolis", "What is the population of Pittsburgh?",
        "How many people live in Denver?", "How big is Austin in square miles?",
        "Which metro is bigger in area, Pittsburgh or Denver", "land area and population density of Miami",
        "population of Portland", "population of Portland, ME", "median household income in Pittsburgh",
        "what counties are in the Denver metro", "census data for Austin", "What are the population, land area and density of Pittsburgh?",
        "population of Fort Worth", "Pittsburgh's population", "land area of Pittsburgh in square kilometers",
    ],
}
