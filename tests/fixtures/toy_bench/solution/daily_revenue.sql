select
    order_date,
    count(*) as num_orders,
    sum(amount) as revenue
from {{ target.schema }}.orders
where status <> 'cancelled'
group by order_date
