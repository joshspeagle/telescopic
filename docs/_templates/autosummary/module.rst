{{ fullname | escape | underline }}

.. automodule:: {{ fullname }}
   :members:
{%- if classes %}
   :exclude-members: {{ classes | join(", ") }}

   .. rubric:: Classes

   .. autosummary::
      :nosignatures:
{% for item in classes %}
      {{ item }}
{%- endfor %}
{%- endif %}
