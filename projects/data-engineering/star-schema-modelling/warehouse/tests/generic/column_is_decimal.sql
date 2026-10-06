{% test column_is_decimal(model, column_name) %}
-- Money must be DECIMAL. A float column passes every not_null, unique and
-- relationships test, so this checks the type itself.
select distinct typeof({{ column_name }}) as column_type
from {{ model }}
where typeof({{ column_name }}) not like 'DECIMAL%'
{% endtest %}
