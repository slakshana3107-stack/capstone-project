-- QUERY 1: Customers with high-value completed orders
SELECT
    customer_id,
    ROUND(SUM(order_amount), 2) AS total_spending
FROM orders
WHERE status = 'completed'
  AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY total_spending DESC;


-- QUERY 2: Revenue by city and product category
SELECT
    c.city,
    p.category,
    ROUND(SUM(o.order_amount), 2) AS total_revenue
FROM orders o
JOIN customers c
    ON o.customer_id = c.customer_id
JOIN products p
    ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY c.city, p.category;


-- QUERY 3: Products priced above average
SELECT
    product_id,
    product_name,
    price_inr
FROM products
WHERE price_inr > (
    SELECT AVG(price_inr)
    FROM products
)
ORDER BY price_inr DESC;


-- QUERY 4: Customers with no completed orders
SELECT
    c.customer_id,
    c.city
FROM customers c
WHERE NOT EXISTS (
    SELECT 1
    FROM orders o
    WHERE o.customer_id = c.customer_id
      AND o.status = 'completed'
)
ORDER BY c.customer_id;


-- QUERY 5: Customers spending above average
WITH customer_spending AS (
    SELECT
        o.customer_id,
        SUM(o.order_amount) AS total_spending
    FROM orders o
    WHERE o.status = 'completed'
    GROUP BY o.customer_id
),
average_spending AS (
    SELECT AVG(total_spending) AS avg_spending
    FROM customer_spending
)
SELECT
    cs.customer_id,
    ROUND(cs.total_spending, 2) AS total_spending,
    ROUND(a.avg_spending, 2) AS average_spending
FROM customer_spending cs
CROSS JOIN average_spending a
WHERE cs.total_spending > a.avg_spending
ORDER BY cs.total_spending DESC;



-- QUERY 6: Rank customers by spending within each city
WITH customer_spending AS (
    SELECT
        c.city,
        c.customer_id,
        ROUND(SUM(o.order_amount), 2) AS total_spending
    FROM customers c
    JOIN orders o
        ON c.customer_id = o.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.city, c.customer_id
),
customer_ranks AS (
    SELECT
        city,
        customer_id,
        total_spending,
        RANK() OVER (
            PARTITION BY city
            ORDER BY total_spending DESC
        ) AS spending_rank,

        DENSE_RANK() OVER (
            PARTITION BY city
            ORDER BY total_spending DESC
        ) AS dense_rank,

        ROW_NUMBER() OVER (
            PARTITION BY city
            ORDER BY total_spending DESC
        ) AS row_num
    FROM customer_spending
)
SELECT
    city,
    customer_id,
    total_spending,
    spending_rank,
    dense_rank,
    row_num
FROM customer_ranks
ORDER BY city, spending_rank, customer_id;


-- QUERY 7: Daily revenue with running total
WITH daily_revenue AS (
    SELECT
        DATE(order_ts) AS order_date,
        SUM(order_amount) AS daily_revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY DATE(order_ts)
)
SELECT
    order_date,
    ROUND(daily_revenue, 2) AS daily_revenue,
    ROUND(
        SUM(daily_revenue) OVER (
            ORDER BY order_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ),
        2
    ) AS running_revenue
FROM daily_revenue
ORDER BY order_date;