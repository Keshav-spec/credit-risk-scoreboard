--   
-- 1. DEFAULT RATE BY SCORE BAND
--   

SELECT
    score_band,
    COUNT(*) AS total_loans,
    SUM(target) AS defaults,
    ROUND(AVG(target) * 100, 2) AS default_rate_pct,
    ROUND(AVG(loan_amnt), 0) AS avg_loan_amount
FROM loan_scores
GROUP BY score_band
ORDER BY default_rate_pct DESC;


--   
-- 2. AVERAGE SCORE BY LOAN GRADE
--   

SELECT
    grade,
    ROUND(AVG(credit_score), 1) AS avg_score,
    COUNT(*) AS loan_count,
    ROUND(AVG(target) * 100, 2) AS actual_default_rate
FROM loan_scores
GROUP BY grade
ORDER BY grade;


--   
-- 3. SCORE DISTRIBUTION SUMMARY
--   

SELECT
    ROUND(MIN(credit_score), 0) AS min_score,
    ROUND(AVG(credit_score), 0) AS avg_score,
    ROUND(MAX(credit_score), 0) AS max_score,

    COUNT(
        CASE
            WHEN credit_score > 650 THEN 1
        END
    ) AS approved_count,

    ROUND(
        COUNT(
            CASE
                WHEN credit_score > 650 THEN 1
            END
        ) * 100.0 / COUNT(*),
        1
    ) AS approval_rate_pct

FROM loan_scores;


--   
-- 4. HIGH-RISK LARGE LOANS
--   

SELECT
    loan_amnt,
    credit_score,
    grade,
    purpose,
    annual_inc,
    dti,
    target

FROM loan_scores

WHERE credit_score < 450
  AND loan_amnt > 20000

ORDER BY loan_amnt DESC

LIMIT 20;

--   
-- 5. DEFAULT RATE BY PURPOSE
--   

SELECT
    purpose,
    COUNT(*) AS loans,
    ROUND(AVG(target) * 100, 2) AS default_rate_pct,
    ROUND(AVG(credit_score), 0) AS avg_score

FROM loan_scores

GROUP BY purpose

HAVING COUNT(*) > 100

ORDER BY default_rate_pct DESC;