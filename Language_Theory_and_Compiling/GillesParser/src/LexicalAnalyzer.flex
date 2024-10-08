%%

%class LexicalAnalyzer
%unicode
%line
%column
%type Symbol
%standalone

// Return value of the program
%eofval{
	return new Symbol(LexicalUnit.EOS, yyline, yycolumn);
%eofval}

// Extended Regular Expressions

AlphaUpperCase = [A-Z]
AlphaLowerCase = [a-z]
Alpha          = {AlphaUpperCase}|{AlphaLowerCase}
Numeric        = [0-9]
AlphaNumeric   = {Alpha}|{Numeric}

Sign           = [+-]
Integer        = {Sign}?(([1-9][0-9]*)|0)
Decimal        = \.[0-9]*
Exponent       = [eE]{Integer}
Real           = {Integer}{Decimal}?{Exponent}?
Identifier     = {Alpha}{AlphaNumeric}*
//comment would be either //comment or !!comment!!
Comment        = (\/\/.*\n)|(\!\!.*\!\!)
//Long comment would be !! comment !!
LongComment   = \!\!.*\!\!

%%// Identification of tokens

// Program structure
"LET"           {System.out.println(new Symbol(LexicalUnit.LET, yyline, yycolumn, yytext()).toString());}
"BE"            {System.out.println(new Symbol(LexicalUnit.BE, yyline, yycolumn, yytext()).toString());}
"END"           {System.out.println(new Symbol(LexicalUnit.END, yyline, yycolumn, yytext()).toString());}
":"             {System.out.println(new Symbol(LexicalUnit.COLUMN, yyline, yycolumn, yytext()).toString());}
"="             {System.out.println(new Symbol(LexicalUnit.ASSIGN, yyline, yycolumn, yytext()).toString());}
"("             {System.out.println(new Symbol(LexicalUnit.LPAREN, yyline, yycolumn, yytext()).toString());}
")"             {System.out.println(new Symbol(LexicalUnit.RPAREN, yyline, yycolumn, yytext()).toString());}

// Arithmetic Operators
"-"             {System.out.println(new Symbol(LexicalUnit.MINUS, yyline, yycolumn, yytext()).toString());}
"+"             {System.out.println(new Symbol(LexicalUnit.PLUS, yyline, yycolumn, yytext()).toString());}
"*"             {System.out.println(new Symbol(LexicalUnit.TIMES, yyline, yycolumn, yytext()).toString());}
"/"             {System.out.println(new Symbol(LexicalUnit.DIVIDE, yyline, yycolumn, yytext()).toString());}

// Conditional and Loop Keywords
"IF"            {System.out.println(new Symbol(LexicalUnit.IF, yyline, yycolumn, yytext()).toString());}
"THEN"          {System.out.println(new Symbol(LexicalUnit.THEN, yyline, yycolumn, yytext()).toString());}
"ELSE"          {System.out.println(new Symbol(LexicalUnit.ELSE, yyline, yycolumn, yytext()).toString());}
"WHILE"         {System.out.println(new Symbol(LexicalUnit.WHILE, yyline, yycolumn, yytext()).toString());}
"REPEAT"        {System.out.println(new Symbol(LexicalUnit.REPEAT, yyline, yycolumn, yytext()).toString());}

// Logical and Comparison Operators
"=="            {System.out.println(new Symbol(LexicalUnit.EQUAL, yyline, yycolumn, yytext()).toString());}
"<="            {System.out.println(new Symbol(LexicalUnit.SMALEQ, yyline, yycolumn, yytext()).toString());}
"<"             {System.out.println(new Symbol(LexicalUnit.SMALLER, yyline, yycolumn, yytext()).toString());}
"->"            {System.out.println(new Symbol(LexicalUnit.IMPLIES, yyline, yycolumn, yytext()).toString());}
"|"             {System.out.println(new Symbol(LexicalUnit.PIPE, yyline, yycolumn, yytext()).toString());}

// Number
{Real}          {System.out.println(new Symbol(LexicalUnit.NUMBER, yyline, yycolumn, yytext()).toString());}

// C99 variable identifier (ProgName and VarName)
{Identifier}    {System.out.println(new Symbol(LexicalUnit.VARNAME, yyline, yycolumn, yytext()).toString());}

{Comment}       {}
{LongComment}  {}
.               {}
