//import java_cup.runtime.*; uncomment if you use CUP

%%// Options of the scanner

%class SampleLexer	//Name
%unicode			//Use unicode
%line				//Use line counter (yyline variable)
%column			//Use character counter by line (yycolumn variable)

//you can use either %cup or %standalone
//   %standalone is for a Scanner which works alone and scan a file
//   %cup is to interact with a CUP parser. In this case, you have to return
//        a Symbol object (defined in the CUP library) for each action.
//        Two constructors:
//                          1. Symbol(int id,int line, int column)
//                          2. Symbol(int id,int line, int column,Object value)
%standalone

////////
//CODE//
////////
%init{//code to execute before scanning
	System.out.println("Initialization!");
%init}

%{//adding Java code (methods, inner classes, ...)
%}

%eof{//code to execute after scanning
   System.out.println("Done!");
%eof}

////////////////////////////////
//Extended Regular Expressions//
////////////////////////////////

LineTerminator = \r|\n|\r\n
InputCharacter = [^\r\n]
WhiteSpace     = {LineTerminator} | [ \t\f]

/* comments */
Comment = {TraditionalComment} | {EndOfLineComment} | {DocumentationComment}
TraditionalComment   = "/*" [^*] ~"*/" | "/*" "*"+ "/"
// Comment can be the last line of the file, without line terminator.
EndOfLineComment     = "//" {InputCharacter}* {LineTerminator}?
DocumentationComment = "/**" {CommentContent} "*"+ "/"
CommentContent       = ( [^*] | \*+ [^/*] )*

Identifier = [:jletter:] [:jletterdigit:]*
DecIntegerLiteral = 0 | [1-9][0-9]*

%state STRING
//////////
//States//
//////////

%state YYINITIAL,PRINT

%%//Identification of tokens and actions

<YYINITIAL>{
    {LineTerminator} {
        System.out.println(); // Print the newline when we encounter it
    }

    {InputCharacter}+ {
        // Print the current line number and the text of the line
        System.out.print((yyline + 1) + ": " + yytext());
    }
}

<PRINT>{
	//{EndOfLine} {yybegin(YYINITIAL);}
	{InputCharacter} {System.out.println(yytext());}
	//.           {System.out.println(yytext());} //we print them explicitly
}
