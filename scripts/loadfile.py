from reportlab.pdfgen import canvas

c = canvas.Canvas("mixed_8.pdf")

# phần text
c.drawString(100, 800, "Text region (clean)")
c.drawString(100, 780, "Moallllllllllllllllllllllllllll...")

# phần image
c.drawImage("hihi6.png", 100, 300, width=400, height=300)

c.save()
