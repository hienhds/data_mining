CREATE TABLE experience_backup AS SELECT * FROM experience;

--
SET SQL_SAFE_UPDATES = 0;
DELETE FROM experience
WHERE id IN (
    SELECT id FROM (
        SELECT e1.id
        FROM experience e1
        JOIN experience e2 
          ON e1.description = e2.description 
          AND e1.id > e2.id
        LIMIT 0,1000
    ) AS temp
);


--
DELETE FROM experience
WHERE job_role IS NULL;

--
ALTER TABLE relations ADD COLUMN embedding LONGBLOB;

--
DELETE FROM experience
WHERE candidate_id NOT IN (SELECT candidate_id FROM candidate);
