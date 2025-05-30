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


-- Tạo bảng nodes
CREATE TABLE nodes (
    id INT  PRIMARY KEY,
    node_name VARCHAR(255),
    node_type VARCHAR(50)
);

-- Tạo bảng relations
CREATE TABLE relations (
    id INT PRIMARY KEY,
    relation_name VARCHAR(255)
);

-- Tạo bảng edges
CREATE TABLE edges (
    id INT PRIMARY KEY,
    head_node_id INT,
    tail_node_id INT,
    relation_id INT,
    attributes JSON,
    FOREIGN KEY (head_node_id) REFERENCES nodes(id),
    FOREIGN KEY (tail_node_id) REFERENCES nodes(id),
    FOREIGN KEY (relation_id) REFERENCES relations(id)
);

--
CREATE TABLE test_edges (
    job_id INT NOT NULL,
    candidate_id INT NOT NULL
);

--
INSERT INTO test_edges (candidate_id, job_id)
SELECT head_node_id , tail_node_id 
FROM edges
WHERE relation_id = 3;