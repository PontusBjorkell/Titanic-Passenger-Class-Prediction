DROP VIEW IF EXISTS vw_class_profile;
DROP VIEW IF EXISTS vw_embarkation_class_mix;
DROP VIEW IF EXISTS vw_family_profile;
DROP VIEW IF EXISTS vw_cabin_profile;

CREATE VIEW vw_class_profile AS
SELECT
    Pclass AS passenger_class,
    COUNT(*) AS passenger_count,
    ROUND(AVG(Age), 2) AS average_age,
    ROUND(AVG(Fare), 2) AS average_fare,
    ROUND(AVG(FamilySize), 2) AS average_family_size,
    ROUND(100.0 * AVG(IsAlone), 2) AS percent_travelling_alone,
    ROUND(100.0 * AVG(HasCabin), 2) AS percent_with_cabin
FROM passengers
GROUP BY Pclass;

CREATE VIEW vw_embarkation_class_mix AS
SELECT
    Embarked AS embarkation_port,
    Pclass AS passenger_class,
    COUNT(*) AS passenger_count,
    ROUND(
        100.0 * COUNT(*) /
        SUM(COUNT(*)) OVER (PARTITION BY Embarked),
        2
    ) AS percentage_within_port
FROM passengers
WHERE Embarked IS NOT NULL
GROUP BY Embarked, Pclass;

CREATE VIEW vw_family_profile AS
SELECT
    FamilySizeGroup AS family_size_group,
    Pclass AS passenger_class,
    COUNT(*) AS passenger_count,
    ROUND(AVG(Fare), 2) AS average_fare
FROM passengers
GROUP BY FamilySizeGroup, Pclass;

CREATE VIEW vw_cabin_profile AS
SELECT
    CabinDeck AS cabin_deck,
    Pclass AS passenger_class,
    COUNT(*) AS passenger_count
FROM passengers
GROUP BY CabinDeck, Pclass;