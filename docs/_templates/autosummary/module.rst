{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}
   :members:
   :show-inheritance:
{%- if fullname == "jaxpint.pta" %}
   :exclude-members: GlobalParams
{%- endif %}
