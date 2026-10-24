-- Optional: run in phpMyAdmin. The app also creates this automatically.
CREATE DATABASE IF NOT EXISTS cat_dog_db CHARACTER SET utf8mb4;
USE cat_dog_db;

CREATE TABLE IF NOT EXISTS predictions (
    id            INT AUTO_INCREMENT PRIMARY KEY,
    filename      VARCHAR(255),
    label         VARCHAR(20)  NOT NULL,
    confidence    FLOAT        NOT NULL,
    cat_prob      FLOAT        NOT NULL,
    dog_prob      FLOAT        NOT NULL,
    model_mode    VARCHAR(30),
    ai_description TEXT,
    correct_label VARCHAR(20) NULL,
    image_data    LONGBLOB,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
