### fights := []piece{}
:= is the short variable declaration operator in Go, which both declares and initializes the variable.
[]piece means "a slice of piece structs". A slice is a dynamic, resizable array in Go.
{} initializes the slice as empty (it contains zero elements).
### Channels
Channels in Go are a built-in feature used for communication between goroutines (lightweight threads). They provide a way to safely send and receive values between concurrent parts of your program.
Key points about channels:
Channels are typed, meaning they carry values of a specific type.
You can send a value into a channel from one goroutine and receive it in another.
Channels help synchronize execution and share data safely without explicit locks.
### Receiver
In Go, the part in parentheses before the function name is called the receiver. It tells Go that this function is a method attached to a specific type.
In your code:
func (u user) march(p piece, publishCh chan<- move) 
(u user) is the receiver. It means this function is a method on the user type.
