create schema if not exists {{ target.schema }};
create or replace table {{ target.schema }}.customers as select * from read_csv_auto('/workspace/data/customers.csv');
create or replace table {{ target.schema }}.orders as select * from read_json_auto('/workspace/data/orders.jsonl');
