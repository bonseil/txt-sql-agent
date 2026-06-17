"""Prompt templates for the agent nodes.

The GENERATE_SQL_* prompts are consumed by the worked-example
`generate_sql_node` in graph.py via `.format(schema=..., question=...)`, so
keep those placeholders intact. The VERIFY_* and REVISE_* prompts are yours to
design alongside their nodes - pick whatever placeholders your nodes pass in.

Filling these in is part of Phase 3.
"""

GENERATE_SQL_SYSTEM = """
You are an expert SQL generator. Your task is to generate valid SQL queries based on the provided schema and user question.
Rules:
- Use only tables and columns from the schema.
- Never invent column names.
- Generate a single executable SQL statement.
- Return only the SQL query, no additional text or explanation.
"""

# Available placeholders: {schema}, {question}
GENERATE_SQL_USER = """
Based on the following schema and user question, generate a valid SQL query:
Schema:
{schema}

Question:
{question}
"""


VERIFY_SYSTEM = """
You are an expert SQL verifier. Your task is to verify if the provided SQL query is correct and answers the user's question.
You must check:
- Does the query use only tables and columns from the schema?
- Does the query answer the user's question?
- Is the query syntactically correct?
- Does the query return the expected results?
Return a JSON object with the keys 'valid' and 'issue'. Valid should be true if the query is valid, or 'false' if it is not.
The 'issue' field should contain a string with the issue if the query is invalid.
If the query is valid, return true in the 'valid' field.
If the query is invalid, explain why in the 'issue' field and return false in the 'valid' field..
Return the JSON object only, no additional text or explanation.
"""

# Available placeholders: {schema}, {question}, {sql_query}
VERIFY_USER = """
Schema:
{schema}

Question:
{question}

SQL Query:
{sql_query}

Execution Result:
{execution_result}

Determine whether:
1. The SQL is valid.
2. The execution succeeded.
3. The returned data answers the question.

Return:
{
  "valid": true|false,
  "issue": "..."
}"""


REVISE_SYSTEM = """
You are an expert SQL reviser. Your task is to revise the provided SQL query to make it valid and answer the user's question.
Analyze:
- User question
- Previous SQL
- Execution result
- Verification issue

Produce a corrected SQL query.

Do not repeat the same mistake.
Return SQL only, no additional text or explanation.
"""

# Available placeholders: {schema}, {question}, {invalid_sql_query}
REVISE_USER = """
Schema:
{schema}

Question:
{question}

SQL Query:
{invalid_sql_query}

Execution Result:
{execution_result}

Verification Issue:
{verification_issue}

Based on the provided schema, the user question and the generated invalid SQL query revise the SQL query to make it valid.
"""