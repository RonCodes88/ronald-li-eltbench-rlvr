select
    c.customer_id,
    c.name,
    count(o.order_id) as num_orders,
    coalesce(sum(o.amount), 0) as total_amount,
    min(o.order_date) as first_order_date
from {{ target.schema }}.customers as c
left join {{ target.schema }}.orders as o
    on o.customer_id = c.customer_id and o.status <> 'cancelled'
group by c.customer_id, c.name
