Here’s a concise guide to help get started with **YAML** syntax, showing how to write YAML files, create objects (mappings), lists, and other data types, plus the most important formatting rules.[cloudbees+2](https://www.cloudbees.com/blog/yaml-tutorial-everything-you-need-get-started)

---

## YAML Syntactical Basics

- **Indentation matters!** Use spaces, never tabs. Consistent indentation (commonly 2 spaces) is required for structure.[learnxinyminutes+1](https://learnxinyminutes.com/yaml/)
- **No tab characters are allowed**. YAML parsers will treat tabs as errors.
## Create Key-Value Pairs (Objects/Mappings)

text

`name: Example age: 25 city: "Boston"`

- Keys always end with a colon (`:`) followed by a space and the value.[gettaurus+1](https://gettaurus.org/docs/YAMLTutorial.md)
    
- Values can be strings (with or without quotes), numbers, booleans.
    

## Create Lists (Sequences)

text

`colors:   - red   - green   - blue`

- Lists start with a dash (`-`) and a space, indented under the key.[gettaurus+1](https://gettaurus.org/docs/YAMLTutorial.md)
    

## Nest Objects and Lists

text

`person:   name: Alice   hobbies:     - volleyball     - painting   address:     city: Boston     zip: 02115`

- Indentation shows nesting. Nest lists inside objects, objects inside lists.[gettaurus](https://gettaurus.org/docs/YAMLTutorial.md)
    

## Main Data Types

|Type|Example|Notes|
|---|---|---|
|String|name: "Tanmay"|Quotes optional/unquoted ok|
|Integer|count: 10||
|Float|pi: 3.1415||
|Boolean|is_student: true|Lowercase true/false only|
|List|colors: [red, green, blue]|Dash syntax or inline list|
|Object|user: {name: "A", age: 21}|Indented or inline format|

## Multi-line Strings

- Use pipe (`|`) for literal block (preserves formatting):
    

text

`description: |   This is a   multi-line string.`

- Use greater-than (`>`) for folded block (converts newlines to spaces):
    

text

`note: >   This text will be   collapsed into a   single line.`

## Comments

- Use `#` for comments:
    

text

`# This is a comment name: Tanmay # End-of-line comment`

---

## Important Syntax Rules & Tips

- **Document markers**: Start a YAML file with `---` (optional), especially if multiple docs in one file.[cloudbees](https://www.cloudbees.com/blog/yaml-tutorial-everything-you-need-get-started)
    
- **Space after colon:** Always put a space after `:`.
    
- **Consistency:** Use the same indentation everywhere—no mixing tabs and spaces.[cloudbees+1](https://www.cloudbees.com/blog/yaml-tutorial-everything-you-need-get-started)
    
- **Lists under objects**: Indent lists under their object key, do not align at root unless top-level list.
    
- **Booleans:** Only lowercase (`true`, `false`)—no caps.
    

---

## Mini Example

text

`person:   name: "Tanmay"   age: 21   student: true   skills:     - Python     - Kubernetes     - React   address:     city: Boston     state: MA   bio: |     Computer science student     Interested in research and coding.   notes: >     Loves volleyball and AI.     Also likes Indian food.`

This shows objects, lists, strings, booleans, nesting, and multi-line strings all in one.[circleci+2](https://circleci.com/blog/what-is-yaml-a-beginner-s-guide/)

---

**YAML is easy to read, flexible, and ideal for configs and data sharing. Always double-check indentation and avoid tabs!**[learnxinyminutes+2](https://learnxinyminutes.com/yaml/)

1. [https://spacelift.io/blog/yaml](https://spacelift.io/blog/yaml)
2. [https://www.cloudbees.com/blog/yaml-tutorial-everything-you-need-get-started](https://www.cloudbees.com/blog/yaml-tutorial-everything-you-need-get-started)
3. [https://www.youtube.com/watch?v=1uFVr15xDGg](https://www.youtube.com/watch?v=1uFVr15xDGg)
4. [https://www.tutorialspoint.com/yaml/index.htm](https://www.tutorialspoint.com/yaml/index.htm)
5. [https://circleci.com/blog/what-is-yaml-a-beginner-s-guide/](https://circleci.com/blog/what-is-yaml-a-beginner-s-guide/)
6. [https://gettaurus.org/docs/YAMLTutorial.md](https://gettaurus.org/docs/YAMLTutorial.md)
7. [https://learnxinyminutes.com/yaml/](https://learnxinyminutes.com/yaml/)
8. [https://www.reddit.com/r/devops/comments/17v7aig/recommended_online_resources_to_learn_yaml_for/](https://www.reddit.com/r/devops/comments/17v7aig/recommended_online_resources_to_learn_yaml_for/)
9. [https://www.redhat.com/en/blog/yaml-beginners](https://www.redhat.com/en/blog/yaml-beginners)
10. [https://docs.ansible.com/ansible/latest/reference_appendices/YAMLSyntax.html](https://docs.ansible.com/ansible/latest/reference_appendices/YAMLSyntax.html)