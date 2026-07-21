-- ============================================================================
-- Titanic Passenger Class Prediction
-- Analytical SQL query collection
--
-- Query format:
--   -- name: machine_readable_query_name
--   -- description: Human-readable explanation
--   SELECT ...;
--
-- scripts/run_analysis.py parses these named query blocks and exports each
-- result to a separate CSV file inside reports/sql/.
-- ============================================================================


-- name: 01_dataset_overview
-- description: Overall dataset size, class coverage, survival rate, and missing-value counts.
SELECT
    COUNT(*) AS passenger_count,
    COUNT(DISTINCT PassengerId) AS unique_passengers,
    COUNT(DISTINCT Pclass) AS passenger_classes,
    ROUND(100.0 * AVG(Survived), 2) AS survival_rate_percent,
    SUM(CASE WHEN Age IS NULL THEN 1 ELSE 0 END) AS missing_age_count,
    SUM(CASE WHEN Cabin IS NULL OR TRIM(Cabin) = '' THEN 1 ELSE 0 END)
        AS missing_cabin_count,
    SUM(CASE WHEN Embarked IS NULL OR TRIM(Embarked) = '' THEN 1 ELSE 0 END)
        AS missing_embarkation_count,
    SUM(CASE WHEN Fare IS NULL THEN 1 ELSE 0 END) AS missing_fare_count
FROM passengers;


-- name: 02_passenger_class_distribution
-- description: Passenger count and share of the dataset for each class.
SELECT
    Pclass AS passenger_class,
    COUNT(*) AS passenger_count,
    ROUND(
        100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
        2
    ) AS dataset_share_percent
FROM passengers
GROUP BY Pclass
ORDER BY Pclass;


-- name: 03_class_profile
-- description: Demographic, fare, family, cabin, and survival profile by passenger class.
SELECT
    passenger_class,
    passenger_count,
    average_age,
    average_fare,
    average_family_size,
    percent_travelling_alone,
    percent_with_cabin,
    ROUND(
        100.0 * AVG(Survived),
        2
    ) AS survival_rate_percent
FROM vw_class_profile AS profile
JOIN passengers AS passenger
    ON passenger.Pclass = profile.passenger_class
GROUP BY
    passenger_class,
    passenger_count,
    average_age,
    average_fare,
    average_family_size,
    percent_travelling_alone,
    percent_with_cabin
ORDER BY passenger_class;


-- name: 04_survival_by_class_and_sex
-- description: Survival outcomes by passenger class and sex.
SELECT
    Pclass AS passenger_class,
    Sex AS sex,
    COUNT(*) AS passenger_count,
    SUM(Survived) AS survivor_count,
    COUNT(*) - SUM(Survived) AS non_survivor_count,
    ROUND(
        100.0 * AVG(Survived),
        2
    ) AS survival_rate_percent
FROM passengers
GROUP BY Pclass, Sex
ORDER BY Pclass, Sex;


-- name: 05_fare_statistics_by_class
-- description: Fare distribution statistics for each passenger class.
SELECT
    Pclass AS passenger_class,
    COUNT(Fare) AS passengers_with_fare,
    ROUND(MIN(Fare), 2) AS minimum_fare,
    ROUND(AVG(Fare), 2) AS average_fare,
    ROUND(MAX(Fare), 2) AS maximum_fare,
    ROUND(
        AVG(Fare * Fare) - AVG(Fare) * AVG(Fare),
        2
    ) AS fare_variance
FROM passengers
GROUP BY Pclass
ORDER BY Pclass;


-- name: 06_age_profile_by_class
-- description: Age coverage and broad age-group composition by passenger class.
SELECT
    Pclass AS passenger_class,
    COUNT(*) AS passenger_count,
    COUNT(Age) AS passengers_with_known_age,
    SUM(CASE WHEN Age IS NULL THEN 1 ELSE 0 END) AS missing_age_count,
    ROUND(AVG(Age), 2) AS average_age,
    ROUND(MIN(Age), 2) AS youngest_age,
    ROUND(MAX(Age), 2) AS oldest_age,
    SUM(CASE WHEN Age < 18 THEN 1 ELSE 0 END) AS child_count,
    SUM(CASE WHEN Age >= 18 AND Age < 60 THEN 1 ELSE 0 END) AS adult_count,
    SUM(CASE WHEN Age >= 60 THEN 1 ELSE 0 END) AS senior_count
FROM passengers
GROUP BY Pclass
ORDER BY Pclass;


-- name: 07_embarkation_class_mix
-- description: Passenger-class composition within each embarkation port.
SELECT
    embarkation_port,
    passenger_class,
    passenger_count,
    percentage_within_port
FROM vw_embarkation_class_mix
ORDER BY embarkation_port, passenger_class;


-- name: 08_embarkation_outcomes
-- description: Fare, class, and survival patterns across embarkation ports.
SELECT
    COALESCE(Embarked, 'Unknown') AS embarkation_port,
    COUNT(*) AS passenger_count,
    ROUND(AVG(Pclass), 2) AS average_passenger_class,
    ROUND(AVG(Fare), 2) AS average_fare,
    ROUND(100.0 * AVG(Survived), 2) AS survival_rate_percent,
    ROUND(100.0 * AVG(HasCabin), 2) AS percent_with_cabin
FROM passengers
GROUP BY COALESCE(Embarked, 'Unknown')
ORDER BY passenger_count DESC;


-- name: 09_family_profile
-- description: Passenger-class and fare patterns across engineered family-size groups.
SELECT
    family_size_group,
    passenger_class,
    passenger_count,
    average_fare
FROM vw_family_profile
ORDER BY
    CASE family_size_group
        WHEN 'Alone' THEN 1
        WHEN 'Small' THEN 2
        WHEN 'Medium' THEN 3
        WHEN 'Large' THEN 4
        ELSE 5
    END,
    passenger_class;


-- name: 10_family_survival_profile
-- description: Survival rate and passenger characteristics by family-size group.
SELECT
    FamilySizeGroup AS family_size_group,
    COUNT(*) AS passenger_count,
    ROUND(AVG(FamilySize), 2) AS average_family_size,
    ROUND(AVG(Fare), 2) AS average_fare,
    ROUND(AVG(Pclass), 2) AS average_passenger_class,
    ROUND(100.0 * AVG(Survived), 2) AS survival_rate_percent,
    ROUND(100.0 * AVG(IsAlone), 2) AS percent_travelling_alone
FROM passengers
GROUP BY FamilySizeGroup
ORDER BY passenger_count DESC;


-- name: 11_cabin_profile
-- description: Passenger counts by cabin deck and passenger class.
SELECT
    cabin_deck,
    passenger_class,
    passenger_count
FROM vw_cabin_profile
ORDER BY cabin_deck, passenger_class;


-- name: 12_cabin_access_and_outcomes
-- description: Class, fare, and survival differences between passengers with and without cabin records.
SELECT
    CASE
        WHEN HasCabin = 1 THEN 'Cabin recorded'
        ELSE 'No cabin recorded'
    END AS cabin_status,
    COUNT(*) AS passenger_count,
    ROUND(AVG(Pclass), 2) AS average_passenger_class,
    ROUND(AVG(Fare), 2) AS average_fare,
    ROUND(AVG(Age), 2) AS average_age,
    ROUND(100.0 * AVG(Survived), 2) AS survival_rate_percent
FROM passengers
GROUP BY HasCabin
ORDER BY HasCabin DESC;


-- name: 13_title_profile
-- description: Passenger-class, demographic, fare, and survival patterns by extracted title.
SELECT
    Title AS passenger_title,
    COUNT(*) AS passenger_count,
    ROUND(AVG(Pclass), 2) AS average_passenger_class,
    ROUND(AVG(Age), 2) AS average_age,
    ROUND(AVG(Fare), 2) AS average_fare,
    ROUND(100.0 * AVG(Survived), 2) AS survival_rate_percent
FROM passengers
GROUP BY Title
ORDER BY passenger_count DESC, passenger_title;


-- name: 14_ticket_prefix_profile
-- description: Most common engineered ticket prefixes and their class associations.
SELECT
    TicketPrefix AS ticket_prefix,
    COUNT(*) AS passenger_count,
    ROUND(AVG(Pclass), 2) AS average_passenger_class,
    ROUND(AVG(Fare), 2) AS average_fare,
    ROUND(100.0 * AVG(Survived), 2) AS survival_rate_percent
FROM passengers
GROUP BY TicketPrefix
HAVING COUNT(*) >= 3
ORDER BY passenger_count DESC, ticket_prefix
LIMIT 20;


-- name: 15_high_fare_passengers
-- description: Twenty passengers with the highest recorded fares.
SELECT
    PassengerId AS passenger_id,
    Name AS passenger_name,
    Pclass AS passenger_class,
    Sex AS sex,
    Age AS age,
    Fare AS fare,
    Cabin AS cabin,
    CabinDeck AS cabin_deck,
    TicketPrefix AS ticket_prefix,
    Survived AS survived
FROM passengers
WHERE Fare IS NOT NULL
ORDER BY Fare DESC, PassengerId
LIMIT 20;


-- name: 16_largest_family_groups
-- description: Passengers belonging to the largest onboard family groups.
SELECT
    PassengerId AS passenger_id,
    Name AS passenger_name,
    Pclass AS passenger_class,
    FamilySize AS family_size,
    FamilySizeGroup AS family_size_group,
    SibSp AS siblings_or_spouses,
    Parch AS parents_or_children,
    Fare AS fare,
    Survived AS survived
FROM passengers
ORDER BY FamilySize DESC, Fare DESC, PassengerId
LIMIT 25;


-- name: 17_feature_summary
-- description: Summary statistics for the principal engineered numeric and binary features.
SELECT
    ROUND(AVG(FamilySize), 3) AS average_family_size,
    MIN(FamilySize) AS minimum_family_size,
    MAX(FamilySize) AS maximum_family_size,
    ROUND(100.0 * AVG(IsAlone), 2) AS percent_alone,
    ROUND(100.0 * AVG(HasCabin), 2) AS percent_with_cabin,
    ROUND(AVG(CabinCount), 3) AS average_cabin_count,
    MAX(CabinCount) AS maximum_cabin_count,
    ROUND(AVG(FareLog), 3) AS average_log_fare,
    ROUND(MIN(FareLog), 3) AS minimum_log_fare,
    ROUND(MAX(FareLog), 3) AS maximum_log_fare
FROM passengers;


-- name: 18_missing_data_by_class
-- description: Missing-value counts and percentages within each passenger class.
SELECT
    Pclass AS passenger_class,
    COUNT(*) AS passenger_count,

    SUM(CASE WHEN Age IS NULL THEN 1 ELSE 0 END) AS missing_age_count,
    ROUND(
        100.0 * SUM(CASE WHEN Age IS NULL THEN 1 ELSE 0 END) / COUNT(*),
        2
    ) AS missing_age_percent,

    SUM(
        CASE
            WHEN Cabin IS NULL OR TRIM(Cabin) = '' THEN 1
            ELSE 0
        END
    ) AS missing_cabin_count,
    ROUND(
        100.0 * SUM(
            CASE
                WHEN Cabin IS NULL OR TRIM(Cabin) = '' THEN 1
                ELSE 0
            END
        ) / COUNT(*),
        2
    ) AS missing_cabin_percent,

    SUM(
        CASE
            WHEN Embarked IS NULL OR TRIM(Embarked) = '' THEN 1
            ELSE 0
        END
    ) AS missing_embarkation_count
FROM passengers
GROUP BY Pclass
ORDER BY Pclass;


-- name: 19_class_prediction_feature_profile
-- description: Average values of key model features by the target passenger class.
SELECT
    Pclass AS passenger_class,
    COUNT(*) AS passenger_count,
    ROUND(AVG(Age), 2) AS average_age,
    ROUND(AVG(Fare), 2) AS average_fare,
    ROUND(AVG(FareLog), 3) AS average_fare_log,
    ROUND(AVG(FamilySize), 2) AS average_family_size,
    ROUND(100.0 * AVG(IsAlone), 2) AS percent_alone,
    ROUND(100.0 * AVG(HasCabin), 2) AS percent_with_cabin,
    ROUND(AVG(CabinCount), 2) AS average_cabin_count
FROM passengers
GROUP BY Pclass
ORDER BY Pclass;


-- name: 20_potential_duplicate_tickets
-- description: Tickets shared by multiple passengers, which often represent travelling groups or families.
SELECT
    Ticket AS ticket,
    COUNT(*) AS passenger_count,
    COUNT(DISTINCT Pclass) AS represented_classes,
    ROUND(AVG(Fare), 2) AS average_fare,
    GROUP_CONCAT(DISTINCT Pclass) AS passenger_classes
FROM passengers
WHERE Ticket IS NOT NULL
  AND TRIM(Ticket) <> ''
GROUP BY Ticket
HAVING COUNT(*) > 1
ORDER BY passenger_count DESC, ticket
LIMIT 30;