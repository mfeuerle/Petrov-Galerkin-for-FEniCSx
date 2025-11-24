{{ fullname }}
{{ underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :no-members:
   :no-inherited-members:
   :no-special-members:

   {% block attributes %}
   {% if attributes %}
   .. rubric:: Attributes

   
   {% for item in all_attributes %}
         {%- if not item.startswith('_') %}
   .. autoattribute:: {{ item }}
         {%- endif -%}
   {%- endfor %}
   {% endif %}
   {% endblock %}

  {% block methods %}
   {% if methods %}
   .. rubric:: Methods

   .. autosummary::
      :toctree: generated/
   {% for item in all_methods %}
      {%- if not item.startswith('_') or item in ['__matmul__', '__rmatmul__', '__mul__', '__rmul__', '__add__', '__radd__', '__sub__', '__rsub__', '__neg__', '__call__'] %}
      ~{{ name }}.{{ item }}
      {%- endif -%}
   {%- endfor %}
   {% endif %}
  {% endblock %}
