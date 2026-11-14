{{ fullname | escape | underline }}

{#- autosummary sorts the subclasses of Warning and Exception into exceptions. #}
{%- set classes = (classes + exceptions) | sort %}

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
