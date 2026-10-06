{% test unique_combination_of_columns(model, combination_of_columns) %}
-- The grain test. A fact's grain is a set of columns, and the test is that no
-- combination of them appears twice. Written here so the project needs no packages.
select
    {{ combination_of_columns | join(', ') }},
    count(*) as row_count
from {{ model }}
group by {{ combination_of_columns | join(', ') }}
having count(*) > 1
{% endtest %}
