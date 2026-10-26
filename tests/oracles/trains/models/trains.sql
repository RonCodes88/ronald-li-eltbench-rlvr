select
    t.id as train_id,
    t.direction,
    case when t.direction = 'east' and count(c.id) < 4 then 1 else 0 end as run_east_and_has_less_than_4_cars,
    case when count(case when c.len = 'long' then 1 end) > 2 then 1 else 0 end as has_more_than_2_long_cars,
    max(case when c.position = 3 and c.sides = 'double' then 1 else 0 end) as has_double_sided_cars_in_3rd_position,
    max(case when c.position = 1 and c.shape = 'rectangle' then 1 else 0 end) as has_rectangular_shaped_head_cars,
    count(case when c.roof = 'none' then 1 end) as number_of_open_roof_car
from {{ target.database }}.{{ target.schema }}.train as t
left join {{ target.database }}.{{ target.schema }}.cars as c on c.train_id = t.id
group by t.id, t.direction
