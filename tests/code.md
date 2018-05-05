---
layout: page
icon: code
title: Code Blocks Test
---

Inline: `inline code`


Fenced (C++):

```cpp
int main() {
  using namespace std;

  cout << "Jekyll is great!";
}
```


Indented (Python):

    def main:
      langs = ['cpp', 'py', 'java', 'ruby']
      for v in l:
        print "%s is a language." % v
{: .language-python}


Without syntax highlighting:

~~~
public static void main() {
  System.out.println("Please don't use this language.");
}
~~~
