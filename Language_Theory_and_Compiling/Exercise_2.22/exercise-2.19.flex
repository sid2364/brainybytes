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

%eof{//code to execute after scanning
// After the last line, check if it contained alphanumeric content
    if (alphanumericCharCount > 0) {
        alphanumericLineCount++; // Increment line count if it contains alphanumeric chars
    }
    // Print final counts
    System.out.println("Total: " + alphanumericCharCount + " alphanumeric characters, " + alphanumericWordCount + " alphanumeric words, " + alphanumericLineCount + " alphanumeric lines");
    System.out.println("Words: " + alphanumericWords);
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

AlphanumericChar = [a-zA-Z0-9]
AlphanumericWord = {AlphanumericChar}+

// Counters
%{
    int alphanumericCharCount = 0; // To count alphanumeric characters
    int alphanumericWordCount = 0;  // To count alphanumeric words
    int alphanumericLineCount = 0;   // To count alphanumeric lines
    java.util.List<String> alphanumericWords = new java.util.ArrayList<>();
%}

%state STRING
//////////
//States//
//////////

%state YYINITIAL,PRINT

%%//Identification of tokens and actions

<YYINITIAL>{
    {LineTerminator} {
            // Exit condition
            if (alphanumericCharCount > 0) {
                alphanumericLineCount++;
            }
        }

        {AlphanumericChar} {
            alphanumericCharCount++; // Count alphanumeric characters
        }

        {AlphanumericWord} {
            alphanumericWordCount++; // Count alphanumeric words
            alphanumericWords.add(yytext());
        }
}
