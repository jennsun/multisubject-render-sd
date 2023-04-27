from livereload import Server
server = Server()
server.watch('app.py')
server.serve(host='localhost', port=8080)
