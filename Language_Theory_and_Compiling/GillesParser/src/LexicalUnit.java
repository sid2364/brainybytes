/**
 * A terminal symbol, a.k.a. a letter in the grammar.
 */
public enum LexicalUnit{
    /** [ProgName] */
    PROGNAME,
    /** [VarName] */
    VARNAME,
    /** [Number] */
    NUMBER,
    /** <code>LET</code> */
    LET,
    /** <code>BE</code> */
    BE,
    /** <code>END</code> */
    END,
    /** <code>:</code> */
    COLUMN,
    /** <code>=</code> */
    ASSIGN,
    /** <code>(</code> */
    LPAREN,
    /** <code>)</code> */
    RPAREN,
    /** <code>-</code> */
    MINUS,
    /** <code>+</code> */
    PLUS,
    /** <code>*</code> */
    TIMES,
    /** <code>/</code> */
    DIVIDE,
    /** <code>IF</code> */
    IF,
    /** <code>THEN</code> */
    THEN,
    /** <code>ELSE</code> */
    ELSE,
    /** <code>{</code> */
    LBRACK,
    /** <code>}</code> */
    RBRACK,
    /** <code>-></code> */
    IMPLIES,
    /** <code>|</code> */
    PIPE,
    /** <code>==</code> */
    EQUAL,
    /** <code>&lt;=</code> */
    SMALEQ,
    /** <code>&lt;</code> */
    SMALLER,
    /** <code>WHILE</code> */
    WHILE,
    /** <code>REPEAT</code> */
    REPEAT,
    /** <code>OUT</code> */
    OUTPUT,
    /** <code>IN</code> */
    INPUT,
    /** End Of Stream */
    EOS, // End of stream
}
